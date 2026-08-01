# Alignment and Uniformity Evaluation Script

This directory contains a script for evaluating alignment and uniformity metrics of sentence embedding models using the [SentenceTransformer](https://www.sbert.net/) library.

## Overview

`eval.py` computes the following metrics for a given sentence embedding model:
- **Alignment**: Measures how close positive sentence pairs are in the embedding space.
- **Uniformity**: Measures how uniformly the embeddings are distributed on the hypersphere.

The script supports configuration via JSON/YAML/TOML files or command-line arguments.

## Usage

```bash
# Compute both alignment and uniformity
python scripts/constract_llm/eval/isotropic/eval.py main --config_file_path=config/constract_llm/eval/isotropic/example.json

# Compute only alignment
python scripts/constract_llm/eval/isotropic/eval.py alignment --config_file_path=config/constract_llm/eval/isotropic/example.json

# Compute only uniformity
python scripts/constract_llm/eval/isotropic/eval.py uniformity --config_file_path=config/constract_llm/eval/isotropic/example.json
```

You can also override config parameters via CLI:

```bash
python scripts/constract_llm/eval/isotropic/eval.py main --model_name_or_path=sentence-transformers/all-MiniLM-L6-v2 --output_dir=output/
```

## Configuration

The script uses a config file compatible with `pydantic.BaseModel` (see `src/mirei/constract_llm/eval/isotropic/config.py`).
Example (JSON):

```json
{
  "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
  "output_dir": "output/",
  "num_examples": 1000,
  "seed": 42,
  "miracl_name": "miracl-ja",
  "miracl_lang": "ja",
  "wiki_name": "wikipedia-ja",
  "wiki_lang": "ja"
}
```

## Output

Results are saved as JSON files under `<output_dir>/alignment_and_uniformity/<model_name_or_path>/`:

- `main` → `result.json` (both metrics)
- `alignment` → `alignment.json`
- `uniformity` → `uniformity.json`

Each file also records the resolved config, the model name, the number of pairs actually used, and a timestamp, so plots can be rebuilt from the JSON alone.

Example output (`result.json`):

```json
{
  "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
  "config": { "...": "resolved CLIConfig fields" },
  "n_positive_pairs": 1000,
  "n_random_pairs": 1000,
  "meta": { "timestamp": 1752969600, "source": "isotropic_eval" },
  "alignment": 0.1234,
  "alignment_sq_distances": [0.1, 0.15, "..."],
  "uniformity": -1.2345,
  "uniformity_sq_distance_histogram": { "counts": [2, 10, "..."], "bin_edges": [0.0, 0.04, "..."] }
}
```

`alignment_sq_distances` holds the per-pair squared L2 distances of the positive pairs. `uniformity_sq_distance_histogram` is a 100-bin histogram over the squared distances of all ordered off-diagonal embedding pairs (each unordered pair is counted twice).
