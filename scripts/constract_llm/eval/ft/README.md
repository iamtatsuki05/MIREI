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
python scripts/constract_llm/eval/ft/eval.py main --config_file_path=config/constract_llm/eval/sentence_model/example.json

# Compute only alignment
python scripts/constract_llm/eval/ft/eval.py alignment --config_file_path=config/constract_llm/eval/sentence_model/example.json

# Compute only uniformity
python scripts/constract_llm/eval/ft/eval.py uniformity --config_file_path=config/constract_llm/eval/sentence_model/example.json
```

You can also override config parameters via CLI:

```bash
python scripts/constract_llm/eval/ft/eval.py main --model_name_or_path=sentence-transformers/all-MiniLM-L6-v2 --output_dir=output/
```

## Configuration

The script uses a config file compatible with `pydantic.BaseModel` (see `src/nlp/constract_llm/eval/sentence_model/config.py`).
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

Results are saved as JSON files under:

```
<output_dir>/alignment_and_uniformity/<model_name_or_path>/result.json
```

Example output:

```json
{
  "alignment": 0.1234,
  "uniformity": 1.2345
}
```
