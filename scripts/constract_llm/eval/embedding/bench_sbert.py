from pathlib import Path
from typing import Any

import fire
from pydantic import BaseModel, Field

from mirei.common.utils.cli_utils import load_cli_config
from mirei.constract_llm.eval.sentence_model.bench_sbert import benchmark_embeddings


class CLIConfig(BaseModel):
    model_name_or_path: str = Field(..., description='Model name or local path for Sentence-Transformers')
    output_dir: str | Path = Field(..., description='Directory to save results')

    # HF Datasets
    dataset_name_or_path: str = Field(..., description='HF Datasets name or local path')
    dataset_split: str = Field('train', description='Dataset split (e.g., train, validation)')
    text_fields: list[str] = Field(['text'], description='Text column names; concatenated into a single text')
    dataset_config_name: str | None = Field(None, description='Dataset config name (if required)')
    dataset_revision: str | None = Field(None, description='Dataset revision/branch (optional)')
    streaming: bool = Field(True, description='Use streaming mode to iterate lazily')

    # Sweep / Runtime
    batch_sizes: list[int] = Field(..., description='Batch size candidates (e.g., [16, 32, 64])')
    seq_lengths: list[int] = Field(..., description='max_seq_length candidates (e.g., [256, 512, 1024])')
    device: str | None = Field(None, description="Target device, e.g., 'cuda', 'cuda:0', 'cpu'")
    dtype: str | None = Field(None, description="One of 'auto'|'float16'|'bfloat16'|'float32'")
    normalize_embeddings: bool = Field(True, description='Whether to L2-normalize embeddings')
    max_use_samples: int | None = Field(None, description='Max number of samples to use')
    monitor_interval_ms: int = Field(100, description='GPU monitor sampling interval [ms]')
    warmup_fraction: float = Field(0.0, description='Warmup ratio in [0,1]; 0 disables')
    show_progress: bool = Field(True, description='Show tqdm progress bars during embedding')

    # Model initialization (similar to run_st.py)
    cache_dir: str | None = Field(None, description='HF cache directory')
    model_revision: str | None = Field(None, description='Model revision/commit id')
    token: str | None = Field(None, description='HF token for private repos')
    trust_remote_code: bool | None = Field(None, description='Allow loading remote code')
    attn_implementation: str | None = Field(None, description='Attention implementation (e.g., flash_attention_2)')
    low_cpu_mem_usage: bool | None = Field(None, description='Enable low-CPU-memory loading')
    use_fast_tokenizer: bool | None = Field(None, description='Use fast tokenizer backend')

    # Extra passthrough kwargs
    model_extra_kwargs: dict[str, Any] | None = Field(None, description='Extra kwargs for model init')
    tokenizer_extra_kwargs: dict[str, Any] | None = Field(None, description='Extra kwargs for tokenizer init')
    common_extra_kwargs: dict[str, Any] | None = Field(None, description='Extra common kwargs (cache_folder, etc.)')


def main(config_file_path: str | Path | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    benchmark_embeddings(
        model_name_or_path=cfg.model_name_or_path,
        dataset_name_or_path=cfg.dataset_name_or_path,
        dataset_split=cfg.dataset_split,
        text_fields=cfg.text_fields,
        batch_sizes=cfg.batch_sizes,
        seq_lengths=cfg.seq_lengths,
        output_dir=cfg.output_dir,
        device=cfg.device,
        dtype=cfg.dtype,
        normalize_embeddings=cfg.normalize_embeddings,
        max_use_samples=cfg.max_use_samples,
        monitor_interval_ms=cfg.monitor_interval_ms,
        warmup_fraction=cfg.warmup_fraction,
        dataset_config_name=cfg.dataset_config_name,
        dataset_revision=cfg.dataset_revision,
        streaming=cfg.streaming,
        show_progress=cfg.show_progress,
        cache_dir=cfg.cache_dir,
        model_revision=cfg.model_revision,
        token=cfg.token,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        use_fast_tokenizer=cfg.use_fast_tokenizer,
        model_extra_kwargs=cfg.model_extra_kwargs,
        tokenizer_extra_kwargs=cfg.tokenizer_extra_kwargs,
        common_extra_kwargs=cfg.common_extra_kwargs,
    )


if __name__ == '__main__':
    fire.Fire(main)
