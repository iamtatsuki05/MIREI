import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import fire
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from pydantic import BaseModel, Field
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from mirei.common.utils.cli_utils import load_cli_config
from mirei.common.utils.file.json import save_as_indented_json
from mirei.constract_llm.eval.analysis.loader import load_backbone
from mirei.constract_llm.eval.analysis.metrics import (
    ablate_dims,
    attention_summary,
    build_model_inputs,
    effective_rank,
    ordered_text_sha256,
    outlier_stats,
    position_contribution,
    spearman,
)
from mirei.constract_llm.eval.sentence_model.dataset import prepare_dataset
from mirei.constract_llm.eval.sentence_model.metric import compute_alignment, compute_uniformity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREFIX_TEXTS = {
    'en': 'Note that the following statement is recorded here for reference purposes. ',
    'ja': '以下の文は参考のためにここに記録されているものです。',
}
NUM_PREFIX_EXAMPLES = 64
KILL_TEST_TOP_KS = (1, 3, 8)
POSITION_NUM_BINS = 10
POOLING_VARIANTS = ('mean', 'last', 'first', 'pos_weighted')


class CLIConfig(BaseModel):
    output_dir: str = Field(..., description='Output directory for analysis.json')
    model_name_or_path: str = Field(..., description='Model name or path to analyze')
    model_revision: str | None = Field(None, description='Model revision (HF) if any')
    language: Literal['en', 'ja'] = Field(..., description='Language of geometry / STS datasets')
    num_examples: int = Field(2000, gt=0, description='Number of positive pairs for geometry analysis')
    num_attention_examples: int = Field(32, gt=0, description='Number of sentences for attention analysis')
    attention_max_tokens: int = Field(128, gt=0, description='Max tokens per sentence for attention analysis')
    batch_size: int = Field(16, gt=0, description='Batch size for forward passes')
    max_seq_length: int = Field(512, gt=0, description='Max sequence length for geometry / STS forward passes')
    seed: int = Field(42, description='Random seed')
    dtype: str = Field('bfloat16', description='Model dtype (bfloat16 / float16 / float32)')
    with_attention: bool = Field(True, description='Run attention summary analysis')
    with_layer_sts: bool = Field(True, description='Run per-layer STS spearman analysis')
    with_kill_test: bool = Field(True, description='Run outlier-dimension ablation (kill) test')
    dump_layer_embeddings: bool = Field(False, description='Dump per-layer z1 mean embeddings as float16 npz')
    sts_max_pairs: int = Field(1500, gt=0, description='Max number of STS pairs')


def _resolve_dtype(dtype: str) -> torch.dtype:
    mapping = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    if dtype not in mapping:
        raise ValueError(f'Unsupported dtype: {dtype!r} (expected one of {sorted(mapping)})')
    return mapping[dtype]


def _prepare_geometry_pairs(cfg: CLIConfig) -> list[tuple[str, str]]:
    if cfg.language == 'ja':
        positive_pairs, _ = prepare_dataset(
            cfg.num_examples,
            miracl_lang='ja',
            wiki_name='wikimedia/wikipedia',
            wiki_lang='20231101.ja',
        )
    else:
        positive_pairs, _ = prepare_dataset(
            cfg.num_examples,
            wiki_name='google/wiki40b',
            wiki_lang='en',
            positive_pair_dataset_name='sentence-transformers/all-nli',
            positive_pair_dataset_config_name='triplet',
            positive_pair_dataset_split='train',
            positive_pair_sentence1_column='anchor',
            positive_pair_sentence2_column='positive',
        )
    return positive_pairs


def _prepare_sts_pairs(cfg: CLIConfig) -> tuple[list[str], list[str], list[float]]:
    if cfg.language == 'en':
        sts_ds = load_dataset('nyu-mll/glue', 'stsb', split='validation')
    else:
        sts_ds = load_dataset('shunk031/JGLUE', name='JSTS', split='validation', trust_remote_code=True)
    sentences1: list[str] = []
    sentences2: list[str] = []
    labels: list[float] = []
    for row in sts_ds:
        if len(sentences1) >= cfg.sts_max_pairs:
            break
        sentences1.append(str(row['sentence1']))
        sentences2.append(str(row['sentence2']))
        labels.append(float(row['label']))
    return sentences1, sentences2, labels


class TokenNormAccumulator:
    """Accumulate token id frequencies and final-layer token embedding norms over a corpus."""

    def __init__(self, vocab_size: int) -> None:
        self.counts = torch.zeros(vocab_size, dtype=torch.float64)
        self.norm_sums = torch.zeros(vocab_size, dtype=torch.float64)

    def update(self, input_ids: torch.Tensor, mask: torch.Tensor, final_hidden: torch.Tensor) -> None:
        valid = mask.bool()
        token_ids = input_ids[valid].cpu()
        token_norms = final_hidden[valid].float().norm(dim=-1).double().cpu()
        self.counts.index_add_(0, token_ids, torch.ones_like(token_norms))
        self.norm_sums.index_add_(0, token_ids, token_norms)

    def summary(self) -> dict[str, Any]:
        observed = self.counts > 0
        frequencies = self.counts[observed]
        mean_norms = self.norm_sums[observed] / frequencies
        rho = spearman(torch.log(frequencies).numpy(), mean_norms.numpy())
        return {
            'spearman_log_freq_vs_norm': float(rho),
            'num_token_types': int(observed.sum()),
            'num_token_occurrences': int(self.counts.sum()),
        }


def _masked_mean(hidden: torch.Tensor, mask_float: torch.Tensor) -> torch.Tensor:
    expanded = mask_float.unsqueeze(-1)
    return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)


def _pooling_variants(final_hidden: torch.Tensor, mask_float: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, seq_len, _ = final_hidden.shape
    batch_indices = torch.arange(batch_size, device=final_hidden.device)
    first_indices = mask_float.argmax(dim=1)
    last_indices = seq_len - 1 - mask_float.flip(1).argmax(dim=1)
    ranks = mask_float.cumsum(dim=1) * mask_float
    weights = ranks / ranks.sum(dim=1, keepdim=True).clamp_min(1.0)
    return {
        'mean': _masked_mean(final_hidden, mask_float),
        'last': final_hidden[batch_indices, last_indices],
        'first': final_hidden[batch_indices, first_indices],
        'pos_weighted': (final_hidden * weights.unsqueeze(-1)).sum(dim=1),
    }


def _encode_layer_means(
    auto_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    device: str,
    batch_size: int,
    max_seq_length: int,
    is_bi_decoder: bool,
    accumulator: TokenNormAccumulator | None = None,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """
    Forward texts in batches and return per-layer masked mean-pooled embeddings (fp32, cpu)
    plus final-layer pooling variants (mean / last / first / pos_weighted).
    """
    layer_chunks: list[list[torch.Tensor]] = []
    variant_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in POOLING_VARIANTS}
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_seq_length, return_tensors='pt')
        encoded = encoded.to(device)
        model_dtype = next(auto_model.parameters()).dtype
        outputs = auto_model(**build_model_inputs(encoded, is_bi_decoder, model_dtype), output_hidden_states=True)
        mask_float = encoded['attention_mask'].to(torch.float32)
        hidden_states = outputs.hidden_states
        if not layer_chunks:
            layer_chunks = [[] for _ in hidden_states]
        for layer_index, hidden in enumerate(hidden_states):
            layer_chunks[layer_index].append(_masked_mean(hidden.float(), mask_float).cpu())
        for name, pooled in _pooling_variants(hidden_states[-1].float(), mask_float).items():
            variant_chunks[name].append(pooled.cpu())
        if accumulator is not None:
            accumulator.update(encoded['input_ids'], encoded['attention_mask'], hidden_states[-1])
    layer_means = [torch.cat(chunks, dim=0) for chunks in layer_chunks]
    final_variants = {name: torch.cat(chunks, dim=0) for name, chunks in variant_chunks.items()}
    return layer_means, final_variants


def _geometry_section(
    z1_layers: list[torch.Tensor],
    z2_layers: list[torch.Tensor],
    z1_variants: dict[str, torch.Tensor],
    z2_variants: dict[str, torch.Tensor],
) -> dict[str, Any]:
    per_layer: dict[str, Any] = {}
    for layer_index, (z1, z2) in enumerate(zip(z1_layers, z2_layers)):
        per_layer[f'layer_{layer_index:02d}'] = {
            'alignment': compute_alignment(z1, z2),
            'uniformity': compute_uniformity(z1),
            'outlier': outlier_stats(z1),
            'effective_rank': effective_rank(z1),
        }
    final_layer_pooling = {
        name: {
            'alignment': compute_alignment(z1_variants[name], z2_variants[name]),
            'uniformity': compute_uniformity(z1_variants[name]),
        }
        for name in POOLING_VARIANTS
    }
    return {'per_layer': per_layer, 'final_layer_pooling': final_layer_pooling}


def _cos_spearman(emb1: torch.Tensor, emb2: torch.Tensor, labels: list[float]) -> float:
    cos = F.cosine_similarity(emb1, emb2, dim=-1)
    return spearman(cos.numpy(), np.asarray(labels))


def _sts_section(
    cfg: CLIConfig,
    s1_layers: list[torch.Tensor],
    s2_layers: list[torch.Tensor],
    s1_variants: dict[str, torch.Tensor],
    s2_variants: dict[str, torch.Tensor],
    labels: list[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {'num_pairs': len(labels)}
    result['pooling_spearman'] = {
        name: _cos_spearman(s1_variants[name], s2_variants[name], labels) for name in POOLING_VARIANTS
    }
    if cfg.with_layer_sts:
        result['per_layer_spearman'] = {
            f'layer_{layer_index:02d}': _cos_spearman(s1, s2, labels)
            for layer_index, (s1, s2) in enumerate(zip(s1_layers, s2_layers))
        }
    return result


def _kill_test_section(
    z1_final: torch.Tensor,
    z2_final: torch.Tensor,
    s1_final: torch.Tensor,
    s2_final: torch.Tensor,
    labels: list[float],
) -> dict[str, Any]:
    max_k = max(KILL_TEST_TOP_KS)
    sts_dims = outlier_stats(torch.cat([s1_final, s2_final], dim=0), top_k=max_k)['top_variance_dims']
    sts_result: dict[str, Any] = {'baseline_spearman': _cos_spearman(s1_final, s2_final, labels)}
    for k in KILL_TEST_TOP_KS:
        dims = sts_dims[:k]
        sts_result[f'top_{k}'] = {
            'ablated_dims': dims,
            'spearman': _cos_spearman(ablate_dims(s1_final, dims), ablate_dims(s2_final, dims), labels),
        }
    geometry_dims = outlier_stats(z1_final, top_k=max_k)['top_variance_dims']
    geometry_result: dict[str, Any] = {
        'baseline': {'alignment': compute_alignment(z1_final, z2_final), 'uniformity': compute_uniformity(z1_final)}
    }
    for k in KILL_TEST_TOP_KS:
        dims = geometry_dims[:k]
        z1_ablated = ablate_dims(z1_final, dims)
        geometry_result[f'top_{k}'] = {
            'ablated_dims': dims,
            'alignment': compute_alignment(z1_ablated, ablate_dims(z2_final, dims)),
            'uniformity': compute_uniformity(z1_ablated),
        }
    return {'sts': sts_result, 'geometry': geometry_result}


def _attention_and_position_sections(
    cfg: CLIConfig,
    auto_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    device: str,
    is_bi_decoder: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    attention_texts = texts[: cfg.num_attention_examples]
    layer_summaries: list[list[dict[str, Any]]] = []
    position_results: list[dict[str, Any]] = []
    for start in range(0, len(attention_texts), cfg.batch_size):
        batch = attention_texts[start : start + cfg.batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=cfg.attention_max_tokens, return_tensors='pt'
        )
        encoded = encoded.to(device)
        model_dtype = next(auto_model.parameters()).dtype
        outputs = auto_model(
            **build_model_inputs(encoded, is_bi_decoder, model_dtype),
            output_attentions=cfg.with_attention,
            output_hidden_states=True,
        )
        mask_batch = encoded['attention_mask'].bool()
        final_hidden = outputs.hidden_states[-1]
        for item_index in range(len(batch)):
            item_mask = mask_batch[item_index]
            if cfg.with_attention:
                if not layer_summaries:
                    layer_summaries = [[] for _ in outputs.attentions]
                for layer_index, layer_attention in enumerate(outputs.attentions):
                    layer_summaries[layer_index].append(attention_summary(layer_attention[item_index], item_mask))
            position_results.append(
                position_contribution(final_hidden[item_index], item_mask, num_bins=POSITION_NUM_BINS)
            )

    attention_section: dict[str, Any] | None = None
    if cfg.with_attention:
        per_layer = []
        for layer_index, summaries in enumerate(layer_summaries):
            mean_metrics = {
                name: float(np.mean([s['mean'][name] for s in summaries])) for name in summaries[0]['mean']
            }
            per_head_metrics = {
                name: np.mean([s['per_head'][name] for s in summaries], axis=0).tolist()
                for name in summaries[0]['per_head']
            }
            per_layer.append({'layer': layer_index, 'mean': mean_metrics, 'per_head': per_head_metrics})
        attention_section = {
            'per_layer': per_layer,
            'num_sentences': len(attention_texts),
            'max_tokens': cfg.attention_max_tokens,
        }

    counts = np.asarray([r['count'] for r in position_results], dtype=np.float64)
    cos_means = np.asarray([r['cos_mean'] for r in position_results], dtype=np.float64)
    shares = np.asarray([r['norm_share'] for r in position_results], dtype=np.float64)
    total_counts = counts.sum(axis=0)
    weighted_cos = np.where(total_counts > 0, (cos_means * counts).sum(axis=0) / np.maximum(total_counts, 1.0), 0.0)
    position_section = {
        'cos_mean': weighted_cos.tolist(),
        'norm_share_mean': shares.mean(axis=0).tolist(),
        'token_counts': total_counts.astype(np.int64).tolist(),
        'num_bins': POSITION_NUM_BINS,
        'num_sentences': len(attention_texts),
    }
    return attention_section, position_section


def _encode_final_mean(
    auto_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    device: str,
    batch_size: int,
    max_seq_length: int,
    is_bi_decoder: bool,
) -> torch.Tensor:
    pooled_chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_seq_length, return_tensors='pt')
        encoded = encoded.to(device)
        model_dtype = next(auto_model.parameters()).dtype
        outputs = auto_model(**build_model_inputs(encoded, is_bi_decoder, model_dtype))
        mask_float = encoded['attention_mask'].to(torch.float32)
        pooled_chunks.append(_masked_mean(outputs.last_hidden_state.float(), mask_float).cpu())
    return torch.cat(pooled_chunks, dim=0)


def _prefix_robustness_section(
    cfg: CLIConfig,
    auto_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    device: str,
    is_bi_decoder: bool,
) -> dict[str, Any]:
    base_texts = texts[:NUM_PREFIX_EXAMPLES]
    prefix = PREFIX_TEXTS[cfg.language]
    prefixed_texts = [prefix + text for text in base_texts]
    base_emb = _encode_final_mean(
        auto_model, tokenizer, base_texts, device, cfg.batch_size, cfg.max_seq_length, is_bi_decoder
    )
    prefixed_emb = _encode_final_mean(
        auto_model, tokenizer, prefixed_texts, device, cfg.batch_size, cfg.max_seq_length, is_bi_decoder
    )
    cos = F.cosine_similarity(base_emb, prefixed_emb, dim=-1)
    return {'mean_cos': float(cos.mean()), 'num_sentences': len(base_texts), 'prefix': prefix}


def _assert_finite(obj: Any, path: str = 'result') -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _assert_finite(value, f'{path}.{key}')
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _assert_finite(value, f'{path}[{index}]')
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f'Non-finite value at {path}: {obj}')


def main(config_file_path: str | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = _resolve_dtype(cfg.dtype)
    auto_model, tokenizer, is_bi_decoder = load_backbone(
        cfg.model_name_or_path, cfg.model_revision, device, torch_dtype
    )

    positive_pairs = _prepare_geometry_pairs(cfg)
    z1_texts = [pair[0] for pair in positive_pairs]
    z2_texts = [pair[1] for pair in positive_pairs]
    sts_s1, sts_s2, sts_labels = _prepare_sts_pairs(cfg)
    logger.info(f'Geometry pairs: {len(positive_pairs)}, STS pairs: {len(sts_labels)}, device: {device}')

    vocab_size = max(int(getattr(auto_model.config, 'vocab_size', 0) or 0), len(tokenizer))
    accumulator = TokenNormAccumulator(vocab_size)

    with torch.inference_mode():
        z1_layers, z1_variants = _encode_layer_means(
            auto_model,
            tokenizer,
            z1_texts,
            device,
            cfg.batch_size,
            cfg.max_seq_length,
            is_bi_decoder,
            accumulator=accumulator,
        )
        z2_layers, z2_variants = _encode_layer_means(
            auto_model,
            tokenizer,
            z2_texts,
            device,
            cfg.batch_size,
            cfg.max_seq_length,
            is_bi_decoder,
            accumulator=accumulator,
        )
        s1_layers, s1_variants = _encode_layer_means(
            auto_model, tokenizer, sts_s1, device, cfg.batch_size, cfg.max_seq_length, is_bi_decoder
        )
        s2_layers, s2_variants = _encode_layer_means(
            auto_model, tokenizer, sts_s2, device, cfg.batch_size, cfg.max_seq_length, is_bi_decoder
        )
        attention_section, position_section = _attention_and_position_sections(
            cfg, auto_model, tokenizer, z1_texts, device, is_bi_decoder
        )
        prefix_section = _prefix_robustness_section(cfg, auto_model, tokenizer, z1_texts, device, is_bi_decoder)

        result: dict[str, Any] = {
            'config': cfg.model_dump(),
            'provenance': {
                'model_name_or_path': cfg.model_name_or_path,
                'model_revision': cfg.model_revision,
                'torch_version': torch.__version__,
                'transformers_version': transformers.__version__,
                'device': device,
                'timestamp': datetime.now(UTC).isoformat(),
            },
            'geometry': _geometry_section(z1_layers, z2_layers, z1_variants, z2_variants),
            'sts': _sts_section(cfg, s1_layers, s2_layers, s1_variants, s2_variants, sts_labels),
        }
        if cfg.with_kill_test:
            result['kill_test'] = _kill_test_section(
                z1_layers[-1], z2_layers[-1], s1_layers[-1], s2_layers[-1], sts_labels
            )
        if attention_section is not None:
            result['attention'] = attention_section
        result['position_contribution'] = position_section
        result['token_frequency_bias'] = accumulator.summary()
        result['prefix_robustness'] = prefix_section

        _assert_finite(result)
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_as_indented_json(result, out_dir / 'analysis.json')
        logger.info(f'Saved analysis result: {out_dir / "analysis.json"}')

        if cfg.dump_layer_embeddings:
            arrays = {
                f'layer_{layer_index:02d}': layer.numpy().astype(np.float16)
                for layer_index, layer in enumerate(z1_layers)
            }
            arrays['text_sha256'] = np.asarray(ordered_text_sha256(z1_texts))
            arrays['language'] = np.asarray(cfg.language)
            np.savez(out_dir / 'layer_embeddings.npz', **arrays)
            logger.info(f'Saved layer embeddings: {out_dir / "layer_embeddings.npz"}')


if __name__ == '__main__':
    fire.Fire(main)
