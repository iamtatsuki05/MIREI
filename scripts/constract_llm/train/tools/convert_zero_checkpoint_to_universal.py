"""Convert a ZeRO-1/2 checkpoint to a DeepSpeed Universal Checkpoint.

Thin wrapper around ``deepspeed.checkpoint.ds_to_universal`` for resuming a
run on a different number of GPUs. The stock converter hardcodes Adam's
``exp_avg``/``exp_avg_sq`` state keys and crashes with ``KeyError: 'exp_avg'``
on our ``schedule_free_radam`` checkpoints (state keys ``z``/``exp_avg_sq``),
so this script surgically rewrites the two hardcoded spots at import time.
The rewrite asserts on the exact upstream source, so a DeepSpeed upgrade that
changes those lines fails loudly instead of converting garbage.

Usage (CPU only; run on a login/CPU node, mind RAM ~= one optimizer shard per
extract worker). Work on a COPY of the checkpoint directory: DeepSpeed resumes
from ``<checkpoint-N>/<tag>`` named by ``<checkpoint-N>/latest_universal``, and
this converter writes ``latest_universal`` into the parent of
``--output_folder`` — so the output must live inside the checkpoint directory
you will resume from, next to its ``trainer_state.json``:

    cp -r /path/to/checkpoint-N /path/to/checkpoint-N-uni
    python convert_zero_checkpoint_to_universal.py \
        --input_folder  /path/to/checkpoint-N-uni/global_stepN \
        --output_folder /path/to/checkpoint-N-uni/global_stepN_universal \
        --inject_missing_state --num_extract_workers 2 --num_merge_workers 1

``--inject_missing_state`` is required for HF Trainer checkpoints (they lack
the ``universal_checkpoint_info`` metadata). After converting, resume once
with ``universal_checkpoint_resume: true`` in the training config (see
``mirei.constract_llm.train.universal_checkpoint``) pointing
``resume_from_checkpoint`` at ``checkpoint-N-uni``, and adjust
``gradient_accumulation_steps`` inversely so the global batch is unchanged,
then remove the flag before any later restart.
"""

import argparse
import inspect
import sys
import textwrap
from pathlib import Path

from mirei.constract_llm.train.universal_checkpoint import detect_optimizer_state_keys

TESTED_DEEPSPEED_SERIES = '0.17.'

# Exact upstream snippets (deepspeed 0.17.5) that hardcode Adam's state keys.
EXTRACT_ORIGINAL = """\
        flat_state = dict(
            exp_avg=state_groups[param_group_id]["exp_avg"],
            exp_avg_sq=state_groups[param_group_id]["exp_avg_sq"],
            fp32=fp32_groups[param_group_id],
        )
"""
EXTRACT_PATCHED = """\
        flat_state = {key: state_groups[param_group_id][key] for key in _MIREI_STATE_KEYS}
        flat_state["fp32"] = fp32_groups[param_group_id]
"""
MERGE_ORIGINAL = 'for state in ("fp32", "exp_avg", "exp_avg_sq"):'
MERGE_PATCHED = 'for state in _MIREI_UNIVERSAL_KEYS:'


def _repatch_function(module, function_name: str, original: str, patched: str) -> None:
    source = textwrap.dedent(inspect.getsource(getattr(module, function_name)))
    if source.count(original) != 1:
        raise RuntimeError(
            f'{function_name} in deepspeed {module.__name__} does not match the expected source '
            f'(found {source.count(original)} occurrences of the patch anchor); '
            'this DeepSpeed version is untested — review the upstream code before converting.'
        )
    # Intentional exec: re-defines the upstream function from its own verified
    # source with a one-anchor replacement, so every other line stays verbatim
    # and version drift fails the assert above instead of silently diverging.
    exec(compile(source.replace(original, patched, 1), f'<patched {function_name}>', 'exec'), module.__dict__)  # noqa: S102


def _detect_state_keys(input_folder: Path) -> list[str]:
    import torch

    zero_files = sorted(input_folder.glob('global_step*/*_optim_states.pt')) or sorted(
        input_folder.glob('*_optim_states.pt')
    )
    if not zero_files:
        raise FileNotFoundError(f'no ZeRO optimizer shard found under {input_folder}')
    # Our own checkpoint, so the pickle payload is trusted; mmap keeps RAM flat.
    shard = torch.load(zero_files[0], map_location='cpu', weights_only=False, mmap=True)
    state_groups = shard['optimizer_state_dict']['base_optimizer_state']['state']
    return detect_optimizer_state_keys(state_groups[0])


def main() -> None:
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument(
        '--state-keys',
        nargs='+',
        default=None,
        help='Per-parameter optimizer state keys (default: auto-detect from the first shard).',
    )
    wrapper_parser.add_argument(
        '--allow-untested-deepspeed',
        action='store_true',
        help=f'Proceed on a DeepSpeed outside the tested {TESTED_DEEPSPEED_SERIES}x series.',
    )
    wrapper_args, passthrough = wrapper_parser.parse_known_args()

    import deepspeed
    from deepspeed.checkpoint import ds_to_universal

    if not deepspeed.__version__.startswith(TESTED_DEEPSPEED_SERIES) and not wrapper_args.allow_untested_deepspeed:
        raise RuntimeError(
            f'deepspeed {deepspeed.__version__} is untested with this wrapper '
            f'(tested: {TESTED_DEEPSPEED_SERIES}x); pass --allow-untested-deepspeed to override.'
        )

    sys.argv = [sys.argv[0], *passthrough]
    ds_args = ds_to_universal.parse_arguments()
    if not ds_args.inject_missing_state:
        print(
            'WARNING: HF Trainer checkpoints lack universal_checkpoint_info; '
            'conversion will likely fail without --inject_missing_state.',
            file=sys.stderr,
        )

    state_keys = wrapper_args.state_keys or _detect_state_keys(Path(ds_args.input_folder))
    print(f'optimizer state keys: {state_keys}')
    # The patched functions reach worker processes by reference (fork start
    # method on Linux); a spawn-based interpreter would re-import the unpatched
    # module and fail loudly with KeyError.
    ds_to_universal.__dict__['_MIREI_STATE_KEYS'] = tuple(state_keys)
    ds_to_universal.__dict__['_MIREI_UNIVERSAL_KEYS'] = ('fp32', *state_keys)
    _repatch_function(ds_to_universal, 'extract_zero_shards', EXTRACT_ORIGINAL, EXTRACT_PATCHED)
    _repatch_function(ds_to_universal, 'merge_tp_slices', MERGE_ORIGINAL, MERGE_PATCHED)

    ds_to_universal.main(ds_args)


if __name__ == '__main__':
    main()
