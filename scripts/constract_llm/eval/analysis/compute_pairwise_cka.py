"""Pairwise layer-wise linear CKA / final-layer mutual-kNN overlap between selected points.

Usage:
    python scripts/constract_llm/eval/analysis/compute_pairwise_cka.py \
        --results_root=/path/to/geometry-analysis/<timestamp>

Reads <results_root>/<point>/layer_embeddings.npz (dumped for final/base points by
run_geometry_analysis.py with dump_layer_embeddings=True) and writes
<results_root>/aggregate/cka.tsv. Both points of a pair must have encoded the same
sentence set (i.e. the same language and geometry-data settings). Layers are matched by
nearest relative depth when the two models differ in depth. CPU only.
"""

import json
import sys
from pathlib import Path

import fire
import numpy as np

PAIRS = [
    ('Sentence-ModernBERT-JP-1B-bs1024@final', 'Sentence-Sarashina-Bi-JP-1B-bs1024@final'),
    ('Sentence-Llama-Bi-JP-1B-bs1024@final', 'Sentence-Sarashina-Bi-JP-1B-bs1024@final'),
    ('Sentence-Sarashina-Bi-JP-1B-PT-bs8192@final', 'Sentence-Sarashina-Bi-JP-1B-bs1024@final'),
    ('sarashina2.2-Bi-1b@base-ja', 'Sentence-Sarashina-Bi-JP-1B-PT-bs8192@final'),
    ('ModernBERT-JP-1B-PT-stage2-bs1024@base-ja', 'Sentence-ModernBERT-JP-1B-bs1024@final'),
    ('Llama-Bi-JP-1B-PT-stage2-bs1024@base-ja', 'Sentence-Llama-Bi-JP-1B-bs1024@final'),
    ('Sentence-ModernBERT-EN-1B-bs1024@final', 'Sentence-Sarashina-Bi-EN-1B-bs1024@final'),
    ('Sentence-Llama-Bi-EN-1B-bs1024@final', 'Sentence-Sarashina-Bi-EN-1B-bs1024@final'),
    ('Sentence-Sarashina-Bi-EN-1B-PT-bs8192@final', 'Sentence-Sarashina-Bi-EN-1B-bs1024@final'),
    ('sarashina2.2-Bi-1b@base-en', 'Sentence-Sarashina-Bi-EN-1B-PT-bs8192@final'),
    ('ModernBERT-EN-1B-PT-stage2-bs1024@base-en', 'Sentence-ModernBERT-EN-1B-bs1024@final'),
    ('Llama-Bi-EN-1B-PT-stage2-bs1024@base-en', 'Sentence-Llama-Bi-EN-1B-bs1024@final'),
]


def _load(results_root: Path, point: str) -> tuple[list[np.ndarray], str | None] | None:
    npz_path = results_root / point / 'layer_embeddings.npz'
    if not npz_path.is_file():
        print('missing npz:', point, file=sys.stderr)
        return None
    data = np.load(npz_path)
    layer_keys = sorted(key for key in data.files if key.startswith('layer_'))
    text_sha256 = str(data['text_sha256']) if 'text_sha256' in data.files else None
    return [data[key].astype(np.float64) for key in layer_keys], text_sha256


def _config_subset(results_root: Path, point: str) -> dict[str, object]:
    analysis = json.loads((results_root / point / 'analysis.json').read_text())
    config = analysis['config']
    return {key: config.get(key) for key in ('language', 'num_examples', 'seed', 'max_seq_length')}


def _verify_pair(
    results_root: Path,
    a_name: str,
    b_name: str,
    a_sha: str | None,
    b_sha: str | None,
    allow_unverified_text: bool,
) -> bool:
    """Verify pair comparability; returns True when the ordered text hash was verified."""
    config_a = _config_subset(results_root, a_name)
    config_b = _config_subset(results_root, b_name)
    if config_a != config_b:
        raise RuntimeError(f'config mismatch for pair {a_name} vs {b_name}: {config_a} != {config_b}')
    if a_sha is not None and b_sha is not None:
        if a_sha != b_sha:
            raise RuntimeError(f'text_sha256 mismatch for pair {a_name} vs {b_name}')
        return True
    if not allow_unverified_text:
        raise RuntimeError(
            f'pair {a_name} vs {b_name} lacks text_sha256 in layer_embeddings.npz (legacy dump). '
            'Re-run the dump or pass --allow_unverified_text=True after confirming both points '
            'used identical geometry-data settings.'
        )
    print(f'WARNING: text correspondence unverified for {a_name} vs {b_name}', file=sys.stderr)
    return False


def _cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0)
    y = y - y.mean(0)
    hsic = np.linalg.norm(x.T @ y, 'fro') ** 2
    denom = np.linalg.norm(x.T @ x, 'fro') * np.linalg.norm(y.T @ y, 'fro')
    return float(hsic / max(denom, 1e-12))


def _knn_overlap(x: np.ndarray, y: np.ndarray, k: int = 10, n: int = 1000) -> float:
    # n limits the sentences used (rows are truncated to the first n); recorded in the TSV.
    x = x[:n] / np.maximum(np.linalg.norm(x[:n], axis=1, keepdims=True), 1e-12)
    y = y[:n] / np.maximum(np.linalg.norm(y[:n], axis=1, keepdims=True), 1e-12)
    sim_x = x @ x.T
    sim_y = y @ y.T
    np.fill_diagonal(sim_x, -np.inf)
    np.fill_diagonal(sim_y, -np.inf)
    knn_x = np.argpartition(-sim_x, k, axis=1)[:, :k]
    knn_y = np.argpartition(-sim_y, k, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(knn_x, knn_y)]))


def main(
    results_root: str,
    allow_partial: bool = False,
    allow_unverified_text: bool = False,
    knn_num_points: int = 1000,
) -> None:
    root = Path(results_root)
    dest = root / 'aggregate'
    dest.mkdir(exist_ok=True)
    rows = ['a\tb\tlayer_frac_a\tlayer_a\tlayer_b\tcka\tknn_overlap_final\tknn_num_points\ttext_verified']
    skipped_pairs: list[str] = []
    for a_name, b_name in PAIRS:
        loaded_a = _load(root, a_name)
        loaded_b = _load(root, b_name)
        if loaded_a is None or loaded_b is None:
            skipped_pairs.append(f'{a_name} vs {b_name}')
            continue
        a, a_sha = loaded_a
        b, b_sha = loaded_b
        text_verified = _verify_pair(root, a_name, b_name, a_sha, b_sha, allow_unverified_text)
        knn_final = _knn_overlap(a[-1], b[-1], n=knn_num_points)
        for i in range(len(a)):
            j = round(i * (len(b) - 1) / max(len(a) - 1, 1))
            value = _cka(a[i], b[j])
            rows.append(
                f'{a_name}\t{b_name}\t{i / max(len(a) - 1, 1):.3f}\t{i}\t{j}\t{value:.4f}'
                f'\t{knn_final:.4f}\t{knn_num_points}\t{text_verified}'
            )
        print(f'{a_name} vs {b_name}: final-layer CKA={_cka(a[-1], b[-1]):.3f} knn={knn_final:.3f}')
    if skipped_pairs and not allow_partial:
        raise RuntimeError(
            'Missing layer_embeddings.npz for pairs (pass --allow_partial=True to accept): ' + '; '.join(skipped_pairs)
        )
    for pair in skipped_pairs:
        print('WARNING (allow_partial): skipped', pair, file=sys.stderr)
    (dest / 'cka.tsv').write_text('\n'.join(rows) + '\n')
    print('wrote', dest / 'cka.tsv')


if __name__ == '__main__':
    fire.Fire(main)
