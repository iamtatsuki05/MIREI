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
python scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-0.5B-PT-stage1.json
```

For multi-GPU training:

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-0.5B-PT-stage1.json
```

### Custom Parameters

For detailed parameter information, refer to the data classes in `src/mirei/constract_llm/train/language_model/mlm/data_class/`.

## Masked Next Token Prediction (MNTP)

The `run_mntp.py` script is used for pre-training models using the Masked Next Token Prediction objective, where the model is trained to predict the next token after a masked token.

### Key Features

- Uses AutoModelForCausalLM architecture
- Configurable LoRA parameters for efficient training

### Usage

```bash
python scripts/constract_llm/train/pt/run_mntp.py config/constract_llm/train/pt/Llama-JP-0.5B-PT-stage1.json
```

For multi-GPU training:

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mntp.py config/constract_llm/train/pt/Llama-JP-0.5B-PT-stage1.json
```

### Custom Parameters

- `mask_token_type`: Type of token to use for masking ('blank', 'eos', or 'mask')
- `data_collator_type`: Type of data collator to use ('default' or 'all_mask')
- `lora`: Whether to use LoRA for training
- `lora_r`: Rank of the LoRA matrices
- `lora_dropout`: Dropout probability for LoRA layers
- `stop_after_n_steps`: Number of steps after which to stop training

For detailed parameter information, refer to the data classes in `src/mirei/constract_llm/train/language_model/mntp/data_class/`.

## Sequence Packing

Both `run_mlm.py` and `run_clm.py` accept the same packing options (see `packing.py` in
`src/mirei/constract_llm/train/language_model/`). Packing keeps document boundaries: attention never crosses a
document, so results match per-document training while GPU utilisation improves for short documents.

| Parameter | Default | Description |
|---|---|---|
| `packing` | `false` | Pack several documents into one row (`trl.data_utils.pack_dataset`). MLM requires `line_by_line: true`. |
| `packing_strategy` | `bfd` | `bfd` keeps documents whole (best-fit decreasing, documents longer than the row are truncated); `wrapped` concatenates and chunks like `group_texts` (no boundaries). |
| `packing_seq_length` | `max_seq_length` | Row length in tokens. |
| `packing_encoder_mode` (MLM) | `auto` | `unpad`: rows are split back into documents and ModernBERT re-packs them with `cu_seqlens` (true packing with `flash_attention_2`); `mask`: 3D block-diagonal attention mask + per-document `position_ids` for BERT-style encoders; `auto` picks `unpad` for ModernBERT and `mask` otherwise. |
| `packing_mask_document_starts` (CLM) | `true` | Exclude the first token of every document from the causal LM loss. |

Decoders receive per-document `position_ids`, the FlashAttention varlen kwargs (`cu_seq_lens_q/k`, `max_length_q/k`)
and no `attention_mask`. `flash_attention_2` uses the varlen kwargs directly; `sdpa` / `eager` rely on transformers
(>= 4.53, torch >= 2.6) deriving the block-diagonal mask from the `position_ids` restarts, which requires
`use_cache=false` (the script sets it, and it is persisted in the saved `config.json`; re-enable it for generation).
Older libraries are rejected at start-up instead of silently training without boundaries. With `wrapped` no
boundary handling happens at all. With packing enabled a "batch" is a number of packed rows, i.e. roughly
`per_device_train_batch_size x packing_seq_length` tokens per micro-batch; `packing_seq_length` must not exceed
`max_seq_length`, and MLM packing cannot be combined with `pad_to_max_length`.

## Configuration Files

Configuration files for pre-training are stored in `config/constract_llm/train/pt/`:

- `Llama-JP-0.5B-PT-stage1.json` - Configuration for stage 1 pre-training of Llama-JP-0.5B
- `Llama-JP-0.5B-PT-stage2.json` - Configuration for stage 2 pre-training of Llama-JP-0.5B
- `Llama-EN-0.5B-PT-stage1.json` - Configuration for stage 1 English pre-training of Llama-0.5B
- `Llama-EN-0.5B-PT-stage2.json` - Configuration for stage 2 English pre-training of Llama-0.5B
- `Llama-JP-1B-PT-stage1.json` / `Llama-JP-1B-PT-stage2.json` - Configuration for intermediate-scale Japanese pre-training of Llama-1B
- `Llama-EN-1B-PT-stage1.json` / `Llama-EN-1B-PT-stage2.json` - Configuration for intermediate-scale English pre-training of Llama-1B
- `Llama-JP-3B-PT-stage1.json` / `Llama-JP-3B-PT-stage2.json` - Configuration for scaled Japanese pre-training of Llama-3B
- `Llama-EN-3B-PT-stage1.json` / `Llama-EN-3B-PT-stage2.json` - Configuration for scaled English pre-training of Llama-3B
- `ModernBERT-JP-0.5B-PT-stage1.json` - Configuration for stage 1 pre-training of ModernBERT-JP-0.5B
- `ModernBERT-JP-0.5B-PT-stage2.json` - Configuration for stage 2 pre-training of ModernBERT-JP-0.5B
- `ModernBERT-EN-0.5B-PT-stage1.json` - Configuration for stage 1 English pre-training of ModernBERT-0.5B
- `ModernBERT-EN-0.5B-PT-stage2.json` - Configuration for stage 2 English pre-training of ModernBERT-0.5B
- `ModernBERT-JP-1B-PT-stage1.json` / `ModernBERT-JP-1B-PT-stage2.json` - Configuration for intermediate-scale Japanese pre-training of ModernBERT-1B
- `ModernBERT-EN-1B-PT-stage1.json` / `ModernBERT-EN-1B-PT-stage2.json` - Configuration for intermediate-scale English pre-training of ModernBERT-1B
- `ModernBERT-JP-3B-PT-stage1.json` / `ModernBERT-JP-3B-PT-stage2.json` - Configuration for scaled Japanese pre-training of ModernBERT-3B
- `ModernBERT-EN-3B-PT-stage1.json` / `ModernBERT-EN-3B-PT-stage2.json` - Configuration for scaled English pre-training of ModernBERT-3B


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

For detailed parameter information, refer to the data classes in `src/mirei/constract_llm/train/language_model/clm/data_class/`.
