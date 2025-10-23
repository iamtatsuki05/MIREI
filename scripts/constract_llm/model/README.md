# Model Initialization Script

English / [日本語](README_JA.md)

This directory contains scripts for model initialization.

## Scripts

### `init_model.py`

Initialise a pre-trained model and prepare it for fine-tuning or evaluation.

**Key features**
- Supports encoder, decoder, seq2seq, and generic architectures.
- Loads checkpoints from the Hugging Face Hub or a local path.
- Saves the hydrated model locally or pushes it to the Hub.
- Allows deterministic runs via explicit seed control.

**Usage**

```bash
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json
```

You can also override configuration file values using command-line arguments:

```bash
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json --model_type=encoder --push_to_hub=False
```

## Configuration File

The configuration file is located at `config/constract_llm/model/init_model/config.json`.

### `save_custom_model.py`

Package a fine-tuned or adapter-based model into a standard layout and optionally publish it to the Hub.

**Key features**
- Supports the custom model families defined in `src/nlp/constract_llm/model/save_custom_model.py` (`CUSTOM_MODEL_CONFIGS`).
- Validates task type (e.g. `mntp`, `st`) and required metadata before exporting.
- Saves the full model and tokenizer to a target directory and can push to a private/public Hub repo.

**Usage**

```bash
python scripts/constract_llm/model/save_custom_model.py config/constract_llm/model/save_custom_model/llama_bi.json
python scripts/constract_llm/model/save_custom_model.py --model_name_or_path path/to/model --custom_model_type llama_bi --task_type mntp --output_dir ./artifacts
```

Configuration examples are provided under `config/constract_llm/model/save_custom_model/`. When `push_to_hub` is enabled, remember to pass `repo_id`.
