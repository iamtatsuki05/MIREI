# Fine-tuning Scripts

English / [日本語](README_JA.md)

This directory contains scripts for fine-tuning language models.

## Available Scripts

- `run_st.py` - Script for Sentence Transformer fine-tuning

## Sentence Transformer

The `run_st.py` script is used for fine-tuning Sentence Transformer models, which are designed to generate meaningful embeddings for sentences that capture semantic similarity.

### Key Features

- Supports training with triplet loss (CachedMultipleNegativesRankingLoss)
- Configurable dataset loading and processing
- Support for multiple datasets and subsets
- Evaluation using triplet evaluator

### Usage

```bash
python scripts/constract_llm/train/ft/run_st.py config/constract_llm/train/ft/Sentence-ModernBERT-JP-1.4B.json
```

For multi-GPU training:

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/ft/run_st.py config/constract_llm/train/ft/Sentence-ModernBERT-JP-1.4B.json
```

### Custom Parameters

- `anchor_column_name`: Column name for anchor sentences
- `positive_column_name`: Column name for positive sentences
- `negative_column_name`: Column name for negative sentences
- `label_column_name`: Column name for labels (used for group_by_label batch sampler)
- `evaluator_type`: Type of evaluator to use (e.g., 'triplet')
- `max_subset_samples`: Maximum number of samples per subset
- `streaming`: Whether to use streaming datasets
- `use_all_subset`: Whether to use all available subsets
- `use_subsets`: List of specific subsets to use
- `max_seq_length`: Maximum sequence length for tokenization

For detailed parameter information, refer to the data classes in `src/nlp/constract_llm/train/st/data_class/`.

### Model Parameters

- `model_name_or_path`: Path to pre-trained model or model identifier from Hugging Face Hub
- `loss_cache_mini_batch_size`: Mini-batch size for loss caching
- `loss_scale`: Scale factor for loss
- `torch_dtype`: Data type for model weights (float16, float32, etc.)
- `attn_implementation`: Attention implementation to use (e.g., 'flash_attention_2')
- `low_cpu_mem_usage`: Whether to use low CPU memory usage when loading the model

## Configuration Files

Configuration files for fine-tuning are stored in `config/constract_llm/train/ft/`:

- `Sentence-Llama-Bi-JP-1.4B-PT.json` - Configuration for fine-tuning pre-trained Llama-Bi-JP-1.4B as a Sentence Transformer
- `Sentence-Llama-Bi-JP-1.4B.json` - Configuration for fine-tuning Llama-Bi-JP-1.4B as a Sentence Transformer
- `Sentence-ModernBERT-JP-1.4B-PT.json` - Configuration for fine-tuning pre-trained ModernBERT-JP-1.4B as a Sentence Transformer
- `Sentence-ModernBERT-JP-1.4B.json` - Configuration for fine-tuning ModernBERT-JP-1.4B as a Sentence Transformer
