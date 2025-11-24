# Scripts

English / [日本語](README_JA.md)

This folder collects automation entry points for the text embedding project. Most scripts wrap functionality from `src/mirei` so that datasets, models, tokenizers, and evaluation jobs can be launched from the command line.

## Directory Overview

- `constract_llm/` – Workflow utilities for building, training, packaging, and evaluating language models.
  - `dataset/` – Cleansing, preprocessing, splitting, and hard negative mining helpers.
  - `tokenizer/` – Commands for training new tokenizers or extending and merging existing ones.
  - `model/` – Tools to initialise base checkpoints or export customised bundles for downstream tasks.
  - `train/` – Launch scripts for language-model pre-training (`pt/`) and Sentence Transformer fine-tuning (`ft/`).
  - `eval/` – Evaluation helpers (embedding benchmarks, isotropy metrics, JMTEB runner, JGLUE wrapper).

## Common Usage Pattern

Every script exposes a [Google Fire](https://github.com/google/python-fire) command. Configuration is usually loaded from `config/constract_llm/...` files (JSON, YAML, TOML) and individual values can be overridden through CLI arguments.

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json --do_deduplicate=False
```

## Configuration File Locations

Representative configuration directories:

- `config/constract_llm/dataset/` – cleanse, preprocess, split jobs.
- `config/constract_llm/tokenizer/` – training, SentencePiece merging, token addition.
- `config/constract_llm/model/` – model initialisation and custom package export.
- `config/constract_llm/train/` – pre-training (`pt/`), fine-tuning (`ft/`), and DeepSpeed configs.
- `config/constract_llm/eval/` – isotropy evaluation samples and task-specific configs.

Copy or adapt these templates before executing the corresponding scripts.
