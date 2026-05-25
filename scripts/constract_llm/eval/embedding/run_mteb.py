import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import fire
from pydantic import BaseModel, Field

from mirei.common.utils.cli_utils import load_cli_config

logger = logging.getLogger(__name__)


class CLIConfig(BaseModel):
    model_name_or_path: str = Field(..., description='Model name or local path for MTEB evaluation')
    output_dir: str | Path = Field(..., description='Directory to save MTEB cache and result summaries')
    tasks: list[str] = Field(..., description='MTEB task names, e.g. ["SciFact", "STSBenchmark"]')

    languages: list[str] | None = Field(['eng'], description='MTEB language filters')
    eval_splits: list[str] | None = Field(None, description='MTEB evaluation splits; None uses task defaults')
    exclusive_language_filter: bool = Field(
        False, description='Only keep exact language matches for multilingual tasks'
    )
    overwrite_strategy: Literal['always', 'never', 'only-missing', 'only-cache'] = Field(
        'only-missing',
        description='MTEB result cache overwrite strategy',
    )
    public_only: bool | None = Field(True, description='Run only public tasks when supported by MTEB')
    raise_error: bool = Field(True, description='Raise immediately when a task fails')
    num_proc: int | None = Field(None, description='Number of processes for MTEB data loading/transforms')

    batch_size: int = Field(32, description='Batch size passed to model.encode')
    normalize_embeddings: bool = Field(True, description='Whether to normalize embeddings during encode')
    show_progress_bar: bool = Field(True, description='Show MTEB and encode progress bars')
    encode_extra_kwargs: dict[str, Any] | None = Field(None, description='Extra kwargs passed to model.encode')

    device: str | None = Field(None, description="Target device, e.g., 'cuda', 'cuda:0', 'cpu'")
    cache_dir: str | Path | None = Field(None, description='HF/SentenceTransformers cache directory')
    model_revision: str | None = Field(None, description='Model revision/commit id')
    token: str | None = Field(None, description='HF token for private repos')
    trust_remote_code: bool | None = Field(None, description='Allow loading remote code')
    attn_implementation: str | None = Field(None, description='Attention implementation, e.g. flash_attention_2')
    model_extra_kwargs: dict[str, Any] | None = Field(None, description='Extra kwargs for model init')
    tokenizer_extra_kwargs: dict[str, Any] | None = Field(None, description='Extra kwargs for tokenizer init')
    use_mteb_get_model: bool = Field(
        False,
        description='Use mteb.get_model instead of SentenceTransformer; useful for leaderboard reference models',
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'model_dump'):
        return _jsonable(value.model_dump(mode='json'))
    if hasattr(value, '__dict__'):
        return _jsonable(vars(value))
    return value


def _load_sentence_model(cfg: CLIConfig) -> Any:
    import mteb
    from sentence_transformers import SentenceTransformer

    if cfg.use_mteb_get_model:
        logger.info(f'Loading model via mteb.get_model: {cfg.model_name_or_path}')
        return mteb.get_model(cfg.model_name_or_path)

    model_kwargs = dict(cfg.model_extra_kwargs or {})
    tokenizer_kwargs = dict(cfg.tokenizer_extra_kwargs or {})
    if cfg.attn_implementation is not None:
        model_kwargs.setdefault('attn_implementation', cfg.attn_implementation)

    kwargs: dict[str, Any] = {
        'device': cfg.device,
        'cache_folder': str(cfg.cache_dir) if cfg.cache_dir is not None else None,
        'revision': cfg.model_revision,
        'token': cfg.token,
        'trust_remote_code': cfg.trust_remote_code,
        'model_kwargs': model_kwargs or None,
        'tokenizer_kwargs': tokenizer_kwargs or None,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.info(f'Loading model via SentenceTransformer: {cfg.model_name_or_path}')
    return SentenceTransformer(cfg.model_name_or_path, **kwargs)


def _summarize_tasks(tasks: Sequence[Any]) -> list[dict[str, Any]]:
    summaries = []
    for task in tasks:
        metadata = task.metadata
        summaries.append(
            {
                'name': metadata.name,
                'type': str(metadata.type),
                'languages': list(getattr(task, 'languages', [])),
                'eval_splits': list(getattr(task, 'eval_splits', [])),
                'hf_subsets': list(getattr(task, 'hf_subsets', [])),
            }
        )
    return summaries


def main(config_file_path: str | Path | None = None, **kwargs: Any) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))

    import mteb

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / 'predictions'
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Selecting MTEB tasks: {cfg.tasks}')
    tasks = mteb.get_tasks(
        tasks=cfg.tasks,
        languages=cfg.languages,
        eval_splits=cfg.eval_splits,
        exclusive_language_filter=cfg.exclusive_language_filter,
    )
    task_summary = _summarize_tasks(tasks)
    with (output_dir / 'task_summary.json').open('w') as f:
        json.dump(task_summary, f, indent=2, ensure_ascii=False)

    model = _load_sentence_model(cfg)
    encode_kwargs = dict(cfg.encode_extra_kwargs or {})
    encode_kwargs.setdefault('batch_size', cfg.batch_size)
    encode_kwargs.setdefault('normalize_embeddings', cfg.normalize_embeddings)
    encode_kwargs.setdefault('show_progress_bar', cfg.show_progress_bar)

    logger.info(f'Running MTEB: model={cfg.model_name_or_path}, tasks={[t["name"] for t in task_summary]}')
    results = mteb.evaluate(
        model,
        tasks=tasks,
        encode_kwargs=encode_kwargs,
        cache=mteb.ResultCache(cache_path=cache_dir),
        overwrite_strategy=cfg.overwrite_strategy,
        prediction_folder=prediction_dir,
        show_progress_bar=cfg.show_progress_bar,
        public_only=cfg.public_only,
        raise_error=cfg.raise_error,
        num_proc=cfg.num_proc,
    )

    with (output_dir / 'result_summary.json').open('w') as f:
        json.dump(_jsonable(results), f, indent=2, ensure_ascii=False, default=str)
    logger.info(f'MTEB summaries saved to {output_dir}')


if __name__ == '__main__':
    fire.Fire(main)
