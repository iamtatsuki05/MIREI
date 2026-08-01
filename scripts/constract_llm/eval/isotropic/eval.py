import logging
import time
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from mirei.common.utils.cli_utils import load_cli_config
from mirei.common.utils.file.json import save_as_indented_json
from mirei.constract_llm.eval.sentence_model.config import CLIConfig
from mirei.constract_llm.eval.sentence_model.dataset import prepare_dataset
from mirei.constract_llm.eval.sentence_model.metric import (
    compute_alignment_sq_distances,
    compute_pairwise_sq_distances,
    compute_uniformity_from_sq_distances,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HISTOGRAM_BINS = 100
EXPECTED_LONG_CONTEXT_LENGTH = 8192


def _build_model_kwargs(dtype: str | None, attn_implementation: str | None) -> dict[str, Any]:
    if dtype is None and attn_implementation is None:
        return {}
    if dtype != 'bfloat16' or attn_implementation != 'flash_attention_2':
        raise ValueError(
            'Isotropic model loading supports only the explicit '
            'dtype=bfloat16 and attn_implementation=flash_attention_2 combination.'
        )
    return {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_2',
    }


def _validate_model_contract(model: SentenceTransformer, dtype: str | None, attn_implementation: str | None) -> None:
    if dtype is None and attn_implementation is None:
        return
    auto_model = getattr(model._first_module(), 'auto_model', None)
    if auto_model is None:
        raise RuntimeError('SentenceTransformer first module does not expose auto_model.')
    reported_backend = getattr(auto_model.config, '_attn_implementation', None)
    if reported_backend != attn_implementation:
        raise RuntimeError(
            f'requested attention backend {attn_implementation!r}, but model reported {reported_backend!r}'
        )
    reported_dtype = str(next(model.parameters()).dtype)
    expected_dtype = str(torch.bfloat16)
    if reported_dtype != expected_dtype:
        raise RuntimeError(f'requested model dtype {expected_dtype!r}, but model reported {reported_dtype!r}')
    reported_max_seq_length = getattr(model, 'max_seq_length', None)
    reported_max_positions = getattr(auto_model.config, 'max_position_embeddings', None)
    if (
        reported_max_seq_length != EXPECTED_LONG_CONTEXT_LENGTH
        or reported_max_positions != EXPECTED_LONG_CONTEXT_LENGTH
    ):
        raise RuntimeError(
            '8192-token benchmark contract mismatch: '
            f'max_seq_length={reported_max_seq_length!r} max_position_embeddings={reported_max_positions!r}'
        )


def setup_and_encode(cfg: CLIConfig):
    """
    Common setup: seed, device, model, dataset, output dir.
    Returns: model, positive_pairs, random_pairs, out_dir
    """
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positive_pairs, random_pairs = prepare_dataset(
        cfg.num_examples,
        cfg.miracl_name,
        cfg.miracl_lang,
        cfg.wiki_name,
        cfg.wiki_lang,
        cfg.positive_pair_dataset_name,
        cfg.positive_pair_dataset_config_name,
        cfg.positive_pair_dataset_split,
        cfg.positive_pair_sentence1_column,
        cfg.positive_pair_sentence2_column,
    )
    model_id = cfg.model_name_or_path
    logger.info(f'Model: {model_id}')
    model_kwargs = _build_model_kwargs(cfg.dtype, cfg.attn_implementation)
    loader_kwargs: dict[str, Any] = {'device': str(device)}
    if model_kwargs:
        loader_kwargs['model_kwargs'] = model_kwargs
    model = SentenceTransformer(model_id, **loader_kwargs)
    _validate_model_contract(model, cfg.dtype, cfg.attn_implementation)
    out_dir = Path(cfg.output_dir) / 'alignment_and_uniformity' / model_id.replace('/', '_')
    out_dir.mkdir(parents=True, exist_ok=True)
    return model, positive_pairs, random_pairs, out_dir


def _base_result(cfg: CLIConfig, n_positive_pairs: int, n_random_pairs: int) -> dict[str, Any]:
    return {
        'model_name_or_path': cfg.model_name_or_path,
        'config': cfg.model_dump(),
        'n_positive_pairs': n_positive_pairs,
        'n_random_pairs': n_random_pairs,
        'meta': {'timestamp': int(time.time()), 'source': 'isotropic_eval'},
    }


def _histogram(sq_distances: torch.Tensor, bins: int = HISTOGRAM_BINS) -> dict[str, list[int] | list[float]]:
    counts, bin_edges = np.histogram(sq_distances.detach().float().cpu().numpy(), bins=bins)
    return {'counts': counts.tolist(), 'bin_edges': bin_edges.tolist()}


def _compute_alignment_result(
    model: SentenceTransformer,
    positive_pairs: list[tuple[str, str]],
    batch_size: int,
) -> dict[str, Any]:
    z1 = model.encode([pair[0] for pair in positive_pairs], batch_size=batch_size, convert_to_tensor=True)
    z2 = model.encode([pair[1] for pair in positive_pairs], batch_size=batch_size, convert_to_tensor=True)
    sq_distances = compute_alignment_sq_distances(z1, z2)
    return {
        'alignment': sq_distances.mean().item(),
        'alignment_sq_distances': sq_distances.detach().float().cpu().tolist(),
    }


def _compute_uniformity_result(
    model: SentenceTransformer,
    random_pairs: list[tuple[str, str]],
    batch_size: int,
) -> dict[str, Any]:
    z1 = model.encode([pair[0] for pair in random_pairs], batch_size=batch_size, convert_to_tensor=True)
    z2 = model.encode([pair[1] for pair in random_pairs], batch_size=batch_size, convert_to_tensor=True)
    sq_distances = compute_pairwise_sq_distances(torch.cat([z1, z2], dim=0))
    return {
        'uniformity': compute_uniformity_from_sq_distances(sq_distances),
        'uniformity_sq_distance_histogram': _histogram(sq_distances),
    }


def alignment(config_file_path: str | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    model, positive_pairs, random_pairs, out_dir = setup_and_encode(cfg)

    result = _base_result(cfg, len(positive_pairs), len(random_pairs))
    result.update(_compute_alignment_result(model, positive_pairs, cfg.batch_size))
    logger.info(f'Alignment:  {result["alignment"]:.4f}')
    save_as_indented_json(result, out_dir / 'alignment.json')


def uniformity(config_file_path: str | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    model, positive_pairs, random_pairs, out_dir = setup_and_encode(cfg)

    result = _base_result(cfg, len(positive_pairs), len(random_pairs))
    result.update(_compute_uniformity_result(model, random_pairs, cfg.batch_size))
    logger.info(f'Uniformity: {result["uniformity"]:.4f}')
    save_as_indented_json(result, out_dir / 'uniformity.json')


def main(config_file_path: str | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    model, positive_pairs, random_pairs, out_dir = setup_and_encode(cfg)

    result = _base_result(cfg, len(positive_pairs), len(random_pairs))
    result.update(_compute_alignment_result(model, positive_pairs, cfg.batch_size))
    result.update(_compute_uniformity_result(model, random_pairs, cfg.batch_size))
    logger.info(f'Alignment:  {result["alignment"]:.4f}')
    logger.info(f'Uniformity: {result["uniformity"]:.4f}')
    save_as_indented_json(result, out_dir / 'result.json')


if __name__ == '__main__':
    fire.Fire({'main': main, 'alignment': alignment, 'uniformity': uniformity})
