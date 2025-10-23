# Language Model Construction Scripts

English / [日本語](README_JA.md)

This directory orchestrates the end-to-end workflow for constructing and operating language models used in the project.

## Directory Structure

- `dataset/` – Dataset preparation utilities
  - `cleanse/` – Data cleansing and heuristic filtering.
  - `preprocess/` – Field-level preprocessing and normalisation.
  - `split/` – Train/validation/test partitioning.
  - `hard_negative_mine/` – Hard negative mining for retrieval and ranking tasks.
- `tokenizer/` – Training, extending, and merging tokenizers.
- `model/` – Initialising base checkpoints and exporting customised model bundles.
- `train/` – Launchers for language-model pre-training (`pt/`) and Sentence Transformer fine-tuning (`ft/`).
- `eval/` – Evaluation helpers:
  - `embedding/bench_sbert.py` – Sentence embedding benchmark runner.
  - `isotropic/eval.py` – Alignment and uniformity (isotropy) metrics.
  - `ft/run_jmteb.sh` – JMTEB evaluation script for sentence transformers.
  - `pt/run_jglue.py` – JGLUE evaluation wrapper for language models.

## Typical Workflow

1. **Prepare datasets** – cleanse, preprocess, split, and optionally mine hard negatives.
2. **Prepare tokenizer** – train a new SentencePiece model or extend/merge existing tokenizers.
3. **Prepare model** – initialise base checkpoints or package custom variants for specific tasks.
4. **Train** – run pre-training jobs (MLM, MNTP, CLM) or Sentence Transformer fine-tuning.
5. **Evaluate** – execute embedding benchmarks, isotropy checks, JMTEB, or JGLUE evaluations.
6. **Publish** – (optional) run `model/save_custom_model.py` to create distributable artefacts or push to the Hub.

Refer to each subdirectory’s README for configuration schemas, command examples, and advanced options.
