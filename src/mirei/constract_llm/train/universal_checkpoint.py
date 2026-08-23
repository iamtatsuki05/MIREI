"""Helpers for resuming DeepSpeed ZeRO checkpoints across world sizes.

DeepSpeed refuses to resume a ZeRO-1/2 checkpoint on a different number of
GPUs because the optimizer state is flat-partitioned by world size. The
official escape hatch is the Universal Checkpoint (UCP) flow:

1. convert the ZeRO checkpoint with ``ds_to_universal`` (see
   ``scripts/constract_llm/train/tools/convert_zero_checkpoint_to_universal.py``,
   which also handles the ``schedule_free_radam`` state keys this project uses;
   the converted ``global_stepN_universal`` folder and the ``latest_universal``
   tag must sit inside the ``checkpoint-N`` directory being resumed),
2. resume once with ``"checkpoint": {"load_universal": true}`` in the
   DeepSpeed config,
3. drop the flag again: checkpoints written after the universal resume are
   regular ZeRO checkpoints, and a leftover ``load_universal`` makes the next
   resume fail at startup.

Step 2 is what this module implements, gated behind the opt-in
``universal_checkpoint_resume`` config key so the default behavior is
untouched.
"""

import json
from pathlib import Path
from typing import Any

UNIVERSAL_CHECKPOINT_RESUME_KEY = 'universal_checkpoint_resume'


def apply_universal_checkpoint_option(config: dict[str, Any]) -> dict[str, Any]:
    """Inject ``load_universal`` into the DeepSpeed config when opted in.

    Must run on the raw CLI config dict *before* ``HfArgumentParser.parse_dict``:
    ``TrainingArguments`` freezes its DeepSpeed configuration during
    ``__post_init__``, so mutating ``training_args.deepspeed`` afterwards has
    no effect. The returned dict carries the DeepSpeed configuration inline
    (as a dict) with ``checkpoint.load_universal = true``.
    """
    if not config.get(UNIVERSAL_CHECKPOINT_RESUME_KEY):
        return config

    deepspeed_config = config.get('deepspeed')
    if not deepspeed_config:
        raise ValueError(
            f'{UNIVERSAL_CHECKPOINT_RESUME_KEY} requires a "deepspeed" config; '
            'universal checkpoint loading is a DeepSpeed feature.'
        )
    if isinstance(deepspeed_config, str):
        deepspeed_config = json.loads(Path(deepspeed_config).read_text())
    elif isinstance(deepspeed_config, dict):
        deepspeed_config = json.loads(json.dumps(deepspeed_config))
    else:
        raise TypeError(f'unsupported "deepspeed" value: {type(deepspeed_config)!r}')

    checkpoint_section = deepspeed_config.setdefault('checkpoint', {})
    checkpoint_section['load_universal'] = True

    updated = dict(config)
    updated['deepspeed'] = deepspeed_config
    return updated


def detect_optimizer_state_keys(state_group: dict[str, Any]) -> list[str]:
    """Return the per-parameter optimizer state keys stored in a ZeRO shard.

    ``ds_to_universal`` hardcodes Adam's ``exp_avg``/``exp_avg_sq``; this
    project trains with ``schedule_free_radam`` whose flat state is
    ``z``/``exp_avg_sq`` instead. ``step`` is excluded because the stock
    converter already handles it separately.
    """
    keys = [key for key in state_group if key != 'step']
    if not keys:
        raise ValueError('optimizer state group holds no per-parameter state')
    return sorted(keys)
