# Training Scripts

English / [日本語](README_JA.md)

This directory provides launch scripts for language-model pre-training and Sentence Transformer fine-tuning.

## Directory Structure

- `pt/` – Pre-training
  - `run_mlm.py` – Masked Language Modeling (encoder-style) training.
  - `run_mntp.py` – Masked Next Token Prediction for causal models with optional LoRA.
  - `run_clm.py` – Causal Language Modeling for decoder-only architectures.
- `ft/` – Sentence Transformer fine-tuning
  - `run_st.py` – Contrastive / triplet-based fine-tuning using Sentence-Transformers Trainer.

## Pre-training

### `run_mlm.py`
- Wraps Hugging Face Transformers’ MLM pipeline (BERT, RoBERTa, ModernBERT, etc.).
- Supports line-by-line and concatenated text processing.
- Compatible with DeepSpeed and DDP setups.

### `run_mntp.py`
- Trains causal models with a masked-next-token objective.
- Provides switches for masking strategy, LoRA rank/dropout, and early stopping.
- Uses `AutoModelForCausalLM` under the hood.

### `run_clm.py`
- Fine-tunes decoder-only models (GPT, Llama, etc.) on causal language modelling tasks.
- Loads Hugging Face `ModelArguments`, `DataTrainingArguments`, and `TrainingArguments` from config.
- Supports streaming datasets, resume-from-checkpoint, and perplexity evaluation.

## Fine-tuning Launcher

### `run_st.py`
- Builds Sentence Transformer training loops with triplet or IR evaluators.
- Supports multiple dataset subsets, streaming, constant-label augmentation, and distributed training.
- Offers rich configuration for anchor/positive/negative columns, evaluators, and model loading options.

## Running the Scripts

All launchers expose [Google Fire](https://github.com/google/python-fire) CLIs that accept JSON/YAML/TOML configs. Basic usage:

```bash
python scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-0.5B-PT-stage1.json
```

For multi-GPU runs, prepend `uv run torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPU` to the command above.

## Configuration Files

Configuration templates live under `config/constract_llm/train/`:

- Pre-training (`pt/`):
  - `Llama-JP-0.5B-PT-stage1.json`
  - `Llama-JP-0.5B-PT-stage2.json`
  - `ModernBERT-JP-0.5B-PT-stage1.json`
  - `ModernBERT-JP-0.5B-PT-stage2.json`
- Fine-tuning (`ft/`):
  - `Sentence-Llama-Bi-JP-0.5B-PT.json`
  - `Sentence-Llama-Bi-JP-0.5B.json`
  - `Sentence-ModernBERT-JP-0.5B-PT.json`
  - `Sentence-ModernBERT-JP-0.5B.json`
  - `Sentence-Sarashina-Bi-0.5B-PT.json`
  - `Sentence-Sarashina-Bi-0.5B.json`
- DeepSpeed configs: `config/constract_llm/train/ds_config/`

Copy a template, adjust hyperparameters, and pass the path to the relevant launch script.
