# Pre-training Scripts

English / [日本語](README_JA.md)

This directory contains scripts for pre-training language models.

## Available Scripts

- `run_mlm.py` - Script for Masked Language Modeling (MLM) pre-training
- `run_mntp.py` - Script for Masked Next Token Prediction (MNTP) pre-training
- `run_clm.py` - Script for Causal Language Modeling (CLM) pre-training and fine-tuning (e.g., GPT, GPT-2, Llama, etc.)

## Masked Language Modeling (MLM)

The `run_mlm.py` script is used for pre-training models using the Masked Language Modeling objective, where random tokens in the input are masked and the model is trained to predict the original tokens.

### Key Features

- Supports various model architectures (BERT, RoBERTa, etc.)
- Configurable masking probability
- Supports both line-by-line and concatenated text processing
- Evaluation using accuracy metrics

### Usage

```bash
python scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-1.4B-PT-stage1.json
```

For multi-GPU training:

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-1.4B-PT-stage1.json
```

### Custom Parameters

For detailed parameter information, refer to the data classes in `src/nlp/constract_llm/train/language_model/mlm/data_class/`.

## Masked Next Token Prediction (MNTP)

The `run_mntp.py` script is used for pre-training models using the Masked Next Token Prediction objective, where the model is trained to predict the next token after a masked token.

### Key Features

- Uses AutoModelForCausalLM architecture
- Configurable LoRA parameters for efficient training

### Usage

```bash
python scripts/constract_llm/train/pt/run_mntp.py config/constract_llm/train/pt/Llama-Bi-JP-1.4B-PT-stage1.json
```

For multi-GPU training:

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mntp.py config/constract_llm/train/pt/Llama-Bi-JP-1.4B-PT-stage1.json
```

### Custom Parameters

- `mask_token_type`: Type of token to use for masking ('blank', 'eos', or 'mask')
- `data_collator_type`: Type of data collator to use ('default' or 'all_mask')
- `lora`: Whether to use LoRA for training
- `lora_r`: Rank of the LoRA matrices
- `lora_dropout`: Dropout probability for LoRA layers
- `stop_after_n_steps`: Number of steps after which to stop training

For detailed parameter information, refer to the data classes in `src/nlp/constract_llm/train/language_model/mntp/data_class/`.

## Configuration Files

Configuration files for pre-training are stored in `config/constract_llm/train/pt/`:

- `Llama-Bi-JP-1.4B-PT-stage1.json` - Configuration for stage 1 pre-training of Llama-Bi-JP-1.4B
- `Llama-Bi-JP-1.4B-PT-stage2.json` - Configuration for stage 2 pre-training of Llama-Bi-JP-1.4B
- `ModernBERT-JP-1.4B-PT-stage1.json` - Configuration for stage 1 pre-training of ModernBERT-JP-1.4B
- `ModernBERT-JP-1.4B-PT-stage2.json` - Configuration for stage 2 pre-training of ModernBERT-JP-1.4B


## Causal Language Modeling (CLM)

The `run_clm.py` script is used for pre-training and fine-tuning models using the Causal Language Modeling objective, where the model is trained to predict the next token in a sequence (auto-regressive). This is suitable for GPT, GPT-2, Llama, and other decoder-based architectures.

### Key Features

- Supports HuggingFace `AutoModelForCausalLM` and compatible architectures
- Flexible dataset loading (local files or HuggingFace datasets)
- Configurable via JSON config files (ModelArguments, DataTrainingArguments, TrainingArguments)
- Supports distributed/multi-GPU training via `torchrun`
- Evaluation with perplexity and accuracy metrics
- Resume training from checkpoints

### Usage

```bash
python scripts/constract_llm/train/pt/run_clm.py config/constract_llm/train/pt/YourCLMConfig.json
```

For multi-GPU training:

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_clm.py config/constract_llm/train/pt/YourCLMConfig.json
```

### Custom Parameters

For detailed parameter information, refer to the data classes in `src/nlp/constract_llm/train/language_model/clm/data_class/`.
