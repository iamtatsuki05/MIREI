import gzip
import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import fire
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from mirei.common.utils.cli_utils import load_cli_config

logger = logging.getLogger(__name__)


class SourceSpec(BaseModel):
    file_name: str = Field(..., description='File name in the Hugging Face dataset repository.')
    label: str = Field(..., description='Source label written to the output JSONL.')
    kind: Literal['pair', 'triplet'] = Field(..., description='Expected row format.')
    max_samples: int | None = Field(None, gt=0, description='Maximum rows to read from this source.')
    random_negative: bool = Field(False, description='Sample negatives from other positives for pair sources.')

    model_config = {'frozen': True}


class CLIConfig(BaseModel):
    mode: Literal['wsl', 'ft'] = Field('wsl', description='Output schema to prepare.')
    repo_id: str = Field('sentence-transformers/embedding-training-data', description='HF dataset repository.')
    revision: str | None = Field(None, description='Dataset repository revision.')
    output_dir: str | Path | None = Field(None, description='Output directory.')
    sources: list[SourceSpec] = Field(..., description='Source files to read from the dataset repository.')
    validation_fraction: float = Field(0.01, ge=0.0, le=1.0, description='Validation split fraction.')
    seed: int = Field(42, description='Random seed for shuffling and negative sampling.')
    max_samples_per_source: int | None = Field(None, gt=0, description='Optional cap applied to each source.')
    cache_dir: str | Path | None = Field(None, description='HF cache directory.')


def _iter_jsonl_gz(path: Path) -> Iterable[Any]:
    with gzip.open(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _as_pair(row: Any) -> tuple[str, str] | None:
    if isinstance(row, list) and len(row) >= 2:
        return str(row[0]), str(row[1])
    if isinstance(row, dict):
        query = row.get('query') or row.get('anchor') or row.get('question')
        positives = row.get('pos') or row.get('positive') or row.get('positives')
        if isinstance(positives, list):
            positive = positives[0] if positives else None
        else:
            positive = positives
        if query and positive:
            return str(query), str(positive)
    return None


def _as_triplet(row: Any) -> tuple[str, str, str] | None:
    if isinstance(row, list) and len(row) >= 3:
        return str(row[0]), str(row[1]), str(row[2])
    if isinstance(row, dict):
        query = row.get('query') or row.get('anchor') or row.get('question')
        positives = row.get('pos') or row.get('positive') or row.get('positives')
        negatives = row.get('neg') or row.get('negative') or row.get('negatives')
        positive = positives[0] if isinstance(positives, list) and positives else positives
        negative = negatives[0] if isinstance(negatives, list) and negatives else negatives
        if query and positive and negative:
            return str(query), str(positive), str(negative)
    return None


def _download_source(
    file_name: str,
    repo_id: str,
    revision: str | None,
    cache_dir: str | Path | None,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            repo_type='dataset',
            revision=revision,
            cache_dir=cache_dir,
        )
    )


def _collect_pairs(
    source: SourceSpec,
    repo_id: str,
    revision: str | None,
    cache_dir: str | Path | None,
) -> list[dict[str, str]]:
    path = _download_source(source.file_name, repo_id, revision, cache_dir)
    rows: list[dict[str, str]] = []
    for row in _iter_jsonl_gz(path):
        pair = _as_pair(row)
        if pair is None:
            continue
        anchor, positive = pair
        if anchor and positive:
            rows.append({'anchor': anchor, 'positive': positive, 'label': source.label})
        if source.max_samples is not None and len(rows) >= source.max_samples:
            break
    return rows


def _collect_triplets(
    source: SourceSpec,
    repo_id: str,
    revision: str | None,
    cache_dir: str | Path | None,
    seed: int,
) -> list[dict[str, str]]:
    if source.kind == 'pair' and source.random_negative:
        pair_rows = _collect_pairs(source, repo_id, revision, cache_dir)
        positives = [row['positive'] for row in pair_rows]
        rng = random.Random(seed)
        triplets = []
        for row in pair_rows:
            if len(positives) < 2:
                break
            negative = positives[rng.randrange(len(positives))]
            while negative == row['positive'] and len(positives) > 1:
                negative = positives[rng.randrange(len(positives))]
            triplets.append(
                {
                    'anchor': row['anchor'],
                    'positive': row['positive'],
                    'negative': negative,
                    'label': source.label,
                }
            )
        return triplets

    path = _download_source(source.file_name, repo_id, revision, cache_dir)
    rows: list[dict[str, str]] = []
    for row in _iter_jsonl_gz(path):
        triplet = _as_triplet(row)
        if triplet is None:
            continue
        anchor, positive, negative = triplet
        if anchor and positive and negative:
            rows.append(
                {
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative,
                    'label': source.label,
                }
            )
        if source.max_samples is not None and len(rows) >= source.max_samples:
            break
    return rows


def _split_rows(
    rows: list[dict[str, str]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    validation_size = max(1, int(len(rows) * validation_fraction)) if rows else 0
    return rows[validation_size:], rows[:validation_size]


def _write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count


def _with_sample_limit(sources: list[SourceSpec], max_samples_per_source: int | None) -> list[SourceSpec]:
    if max_samples_per_source is None:
        return sources
    return [source.model_copy(update={'max_samples': max_samples_per_source}) for source in sources]


def main(config_file_path: str | Path | None = None, **kwargs: Any) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))

    rows: list[dict[str, str]] = []
    sources = _with_sample_limit(cfg.sources, cfg.max_samples_per_source)
    for source in sources:
        logger.info(f'Loading {source.file_name} as {source.kind} source label={source.label}')
        source_rows = (
            _collect_pairs(source, cfg.repo_id, cfg.revision, cfg.cache_dir)
            if cfg.mode == 'wsl'
            else _collect_triplets(source, cfg.repo_id, cfg.revision, cfg.cache_dir, cfg.seed)
        )
        logger.info(f'Loaded {len(source_rows)} rows from {source.file_name}')
        rows.extend(source_rows)

    output_dir = Path(cfg.output_dir or f'data/processed/english_embedding/{cfg.mode}')
    train_rows, validation_rows = _split_rows(rows, cfg.validation_fraction, cfg.seed)
    train_count = _write_jsonl(output_dir / 'train.jsonl', train_rows)
    validation_count = _write_jsonl(output_dir / 'validation.jsonl', validation_rows)

    manifest = {
        'mode': cfg.mode,
        'repo_id': cfg.repo_id,
        'revision': cfg.revision,
        'validation_fraction': cfg.validation_fraction,
        'seed': cfg.seed,
        'max_samples_per_source': cfg.max_samples_per_source,
        'train_rows': train_count,
        'validation_rows': validation_count,
        'sources': [source.model_dump() for source in sources],
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    logger.info(f'Wrote {train_count} train rows and {validation_count} validation rows to {output_dir}')


if __name__ == '__main__':
    fire.Fire(main)
