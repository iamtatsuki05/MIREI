# Training Scripts

English / [日本語](README_JA.md)

This directory contains scripts for training language models.

## Directory Structure

- `pt/`: Scripts for pre-training language models
  - `run_mlm.py` - Script for Masked Language Modeling (MLM) pre-training
  - `run_mntp.py` - Script for Masked Next Token Prediction (MNTP) pre-training
- `ft/`: Scripts for fine-tuning language models
  - `run_st.py` - Script for Sentence Transformer fine-tuning

## Pre-training Scripts

### Masked Language Modeling (MLM)

- `run_mlm.py` - Script for pre-training models using the Masked Language Modeling objective.
  - Supports various model architectures (BERT, RoBERTa, etc.)
  - Configurable masking probability
  - Supports both line-by-line and concatenated text processing
  - Evaluation using accuracy metrics
  - For detailed parameter information, refer to [Hugging Face MLM documentation](https://huggingface.co/docs/transformers/tasks/masked_language_modeling)

### Masked Next Token Prediction (MNTP)

- `run_mntp.py` - Script for pre-training models using the Masked Next Token Prediction objective.
  - Uses AutoModelForCausalLM architecture
  - Supports various masking strategies
  - Configurable LoRA parameters for efficient training
  - For detailed parameter information, refer to [Hugging Face Causal LM documentation](https://huggingface.co/docs/transformers/tasks/language_modeling)

## Fine-tuning Scripts

### Sentence Transformer

- `run_st.py` - Script for fine-tuning Sentence Transformer models.
  - Supports training with triplet loss
  - Configurable dataset loading and processing
  - Evaluation using triplet evaluator
  - Custom parameters:
    - `anchor_column_name`: Column name for anchor sentences
    - `positive_column_name`: Column name for positive sentences
    - `negative_column_name`: Column name for negative sentences
    - `evaluator_type`: Type of evaluator to use (e.g., 'triplet')
    - `max_subset_samples`: Maximum number of samples per subset
    - `streaming`: Whether to use streaming datasets

## Usage

Each script provides a CLI interface using [Google Fire](https://github.com/google/python-fire). Basic usage is as follows:

```bash
python scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-1.4B-PT-stage1.json
```

### Multi-GPU Training

All training scripts support multi-GPU training using PyTorch Distributed Data Parallel (DDP). To run training on multiple GPUs, use the following command format:

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-1.4B-PT-stage1.json
```

Where `$NUM_GPU` is the number of GPUs you want to use for training.

## Configuration Files

Configuration files for each script are stored in the corresponding directories under `config/constract_llm/train/`:

- Pre-training configurations: `config/constract_llm/train/pt/`
  - `Llama-Bi-JP-1.4B-PT-stage1.json`
  - `Llama-Bi-JP-1.4B-PT-stage2.json`
  - `ModernBERT-JP-1.4B-PT-stage1.json`
  - `ModernBERT-JP-1.4B-PT-stage2.json`
- Fine-tuning configurations: `config/constract_llm/train/ft/`
  - `Sentence-Llama-Bi-JP-1.4B-PT.json`
  - `Sentence-Llama-Bi-JP-1.4B.json`
  - `Sentence-ModernBERT-JP-1.4B-PT.json`
  - `Sentence-ModernBERT-JP-1.4B.json`

Each configuration file contains all parameters used by the corresponding script.
