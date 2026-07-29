"""Aggregate per-point analysis.json outputs into scalars.tsv / layers.tsv.

Usage:
    python scripts/constract_llm/eval/analysis/aggregate_results.py \
        --results_root=/path/to/geometry-analysis/<timestamp>

Reads <results_root>/<point>/analysis.json for every point directory and writes
<results_root>/aggregate/{scalars,layers}.tsv. Point directory names follow
`<run>@ckpt-<step>` / `<run>@final` / `<repo>@base[-<lang>]`. CPU only.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

import fire

POINT_RE = re.compile(r'^(?P<run>.+?)@(?P<tag>ckpt-(?P<step>\d+)|final|base(?:-(?P<blang>en|ja))?)$')
KILL_TOP_KS = (1, 3, 8)
NUM_POSITION_BINS = 10


def _classify(run: str, tag: str) -> tuple[str, str]:
    if tag.startswith('base'):
        return 'base', 'base'
    stage = 'wsl' if run.startswith('Sentence-') and run.endswith('-PT') else 'ft'
    if not run.startswith('Sentence-'):
        stage = 'base'
    return stage, ('final' if tag == 'final' else 'ckpt')


def _scalar_row(name: str, run: str, stage: str, kind: str, step: str, analysis: dict[str, Any]) -> dict[str, Any]:
    layers = analysis['geometry']['per_layer']
    last = layers[f'layer_{len(layers) - 1:02d}']
    sts = analysis['sts']
    row: dict[str, Any] = {
        'point': name,
        'run': run,
        'stage': stage,
        'kind': kind,
        'step': step,
        'lang': analysis['config']['language'],
        'n_layers': len(layers),
        'alignment_final': last['alignment'],
        'uniformity_final': last['uniformity'],
        'rankme_final': last['effective_rank']['rankme'],
        'participation_ratio_final': last['effective_rank']['participation_ratio'],
        'top1_var_share_final': last['outlier']['top1_variance_share'],
        'top8_var_share_final': last['outlier']['topk_variance_share'],
        'sts_mean': sts['pooling_spearman']['mean'],
        'sts_last': sts['pooling_spearman']['last'],
        'sts_first': sts['pooling_spearman']['first'],
        'sts_pos_weighted': sts['pooling_spearman']['pos_weighted'],
        'prefix_cos': analysis['prefix_robustness']['mean_cos'],
        'freq_norm_rho': analysis['token_frequency_bias']['spearman_log_freq_vs_norm'],
    }
    attention = analysis.get('attention')
    if attention:
        per_layer = attention['per_layer']
        row['attn_backward_last'] = per_layer[-1]['mean']['backward_mass']
        row['attn_backward_mean'] = sum(p['mean']['backward_mass'] for p in per_layer) / len(per_layer)
        row['attn_backward_max'] = max(p['mean']['backward_mass'] for p in per_layer)
        row['attn_sink_last'] = per_layer[-1]['mean']['sink_mass']
        row['attn_sink_max'] = max(p['mean']['sink_mass'] for p in per_layer)
        row['attn_entropy_last'] = per_layer[-1]['mean']['entropy']
        row['attn_dist_last'] = per_layer[-1]['mean']['mean_distance']
    if 'per_layer_spearman' in sts:
        values = list(sts['per_layer_spearman'].values())
        row['sts_layer_best'] = max(values)
        row['sts_layer_best_idx'] = values.index(max(values))
    kill = analysis.get('kill_test')
    if kill:
        row['kill_sts_base'] = kill['sts']['baseline_spearman']
        for k in KILL_TOP_KS:
            row[f'kill_sts_top{k}'] = kill['sts'][f'top_{k}']['spearman']
    position = analysis['position_contribution']
    for i in range(NUM_POSITION_BINS):
        row[f'pos_cos_bin{i}'] = position['cos_mean'][i]
        row[f'pos_share_bin{i}'] = position['norm_share_mean'][i]
    return row


def _layer_rows(
    name: str, run: str, stage: str, kind: str, step: str, analysis: dict[str, Any]
) -> list[dict[str, Any]]:
    layers = analysis['geometry']['per_layer']
    attention = analysis.get('attention')
    sts_layers = analysis['sts'].get('per_layer_spearman', {})
    rows = []
    for li in range(len(layers)):
        record = layers[f'layer_{li:02d}']
        row: dict[str, Any] = {
            'point': name,
            'run': run,
            'stage': stage,
            'kind': kind,
            'step': step,
            'lang': analysis['config']['language'],
            'layer': li,
            'alignment': record['alignment'],
            'uniformity': record['uniformity'],
            'rankme': record['effective_rank']['rankme'],
            'top1_var_share': record['outlier']['top1_variance_share'],
        }
        if attention and li < len(attention['per_layer']) and kind in ('final', 'base'):
            row.update({f'attn_{k}': v for k, v in attention['per_layer'][li]['mean'].items()})
        if f'layer_{li:02d}' in sts_layers:
            row['sts_spearman'] = sts_layers[f'layer_{li:02d}']
        rows.append(row)
    return rows


def _write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open('w') as fh:
        fh.write('\t'.join(columns) + '\n')
        for row in rows:
            fh.write('\t'.join(str(row.get(column, '')) for column in columns) + '\n')


def main(results_root: str, expected_points: int | None = None, allow_partial: bool = False) -> None:
    out = Path(results_root)
    dest = out / 'aggregate'
    dest.mkdir(exist_ok=True)
    scalar_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    unparsed: list[str] = []
    for point_dir in sorted(out.iterdir()):
        if not point_dir.is_dir() or point_dir.name == 'aggregate':
            continue
        match = POINT_RE.match(point_dir.name)
        if not match:
            unparsed.append(point_dir.name)
            continue
        analysis_file = point_dir / 'analysis.json'
        if not analysis_file.is_file():
            missing.append(point_dir.name)
            continue
        analysis = json.loads(analysis_file.read_text())
        run = match.group('run')
        step = match.group('step') or ('final' if match.group('tag') == 'final' else '0')
        stage, kind = _classify(run, match.group('tag'))
        scalar_rows.append(_scalar_row(point_dir.name, run, stage, kind, step, analysis))
        layer_rows.extend(_layer_rows(point_dir.name, run, stage, kind, step, analysis))
    problems: list[str] = []
    if missing:
        problems.append(f'{len(missing)} point dirs without analysis.json: {missing[:5]}...')
    if unparsed:
        problems.append(f'{len(unparsed)} unparsable point dirs: {unparsed[:5]}...')
    if expected_points is not None and len(scalar_rows) != expected_points:
        problems.append(f'aggregated {len(scalar_rows)} points but expected {expected_points}')
    if problems and not allow_partial:
        raise RuntimeError('Incomplete aggregation (pass --allow_partial=True to accept): ' + ' / '.join(problems))
    for problem in problems:
        print('WARNING (allow_partial):', problem, file=sys.stderr)
    _write_tsv(scalar_rows, dest / 'scalars.tsv')
    _write_tsv(layer_rows, dest / 'layers.tsv')
    print('points aggregated:', len(scalar_rows))
    print('layer rows:', len(layer_rows))
    print('wrote', dest / 'scalars.tsv', dest / 'layers.tsv')


if __name__ == '__main__':
    fire.Fire(main)
