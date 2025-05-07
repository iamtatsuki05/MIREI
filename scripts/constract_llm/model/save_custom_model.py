#!/usr/bin/env python3
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import fire
import transformers
from peft import PeftConfig, PeftModel, get_peft_model
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    PreTrainedTokenizer,
)

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.model.custom.modeling_bidirectional_llama import (
    LlamaBiForMNTP,
    LlamaBiForSequenceClassification,
    LlamaBiModel,
)
from nlp.env import PACKAGE_DIR

CUSTOM_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    'llama': {
        'config_class': LlamaConfig,
        'base_class': LlamaBiModel,
        'mntp_class': LlamaBiForMNTP,
        'seq_class': LlamaBiForSequenceClassification,
        'modeling_py_path': PACKAGE_DIR / 'src/nlp/constract_llm/model/custom/modeling_bidirectional_llama.py',
    },
}

TASK_TYPES: list[str] = ['lm', 'mntp', 'classification']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_transformers_class(cls: type) -> bool:
    if not inspect.isclass(cls):
        return False
    module = inspect.getmodule(cls)
    return bool(module and module.__name__.startswith(transformers.__name__ + '.'))


def set_auto_model_classes(cutom_model_type: str) -> None:
    if cutom_model_type not in CUSTOM_MODEL_CONFIGS:
        raise ValueError(f"Invalid custom_model_type '{cutom_model_type}'.")

    cfg = CUSTOM_MODEL_CONFIGS[cutom_model_type]
    config_cls = cfg['config_class']
    base_cls = cfg['base_class']
    mntp_cls = cfg['mntp_class']
    seq_cls = cfg['seq_class']

    if not is_transformers_class(config_cls) and config_cls is not None:
        config_cls.register_for_auto_class('AutoConfig')
    if not is_transformers_class(base_cls) and base_cls is not None:
        base_cls.register_for_auto_class('AutoModel')
    if not is_transformers_class(mntp_cls) and mntp_cls is not None:
        mntp_cls.register_for_auto_class('AutoModelForCausalLM')
    if not is_transformers_class(seq_cls) and seq_cls is not None:
        seq_cls.register_for_auto_class('AutoModelForSequenceClassification')


def load_custom_model(
    model_name_or_path: str | Path,
    custom_model_type: str,
    task_type: str = 'mntp',
    peft_weights_path: str | Path | None = None,
    peft_config: PeftConfig | None = None,
) -> tuple[Any, PreTrainedTokenizer]:
    if custom_model_type not in CUSTOM_MODEL_CONFIGS:
        raise ValueError(f"Invalid custom_model_type '{custom_model_type}'.")

    set_auto_model_classes(custom_model_type)

    cfg = CUSTOM_MODEL_CONFIGS[custom_model_type]
    model_path = Path(model_name_or_path)

    config = cfg['config_class'].from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if task_type == 'lm':
        ModelClass = cfg['base_class']
    elif task_type == 'mntp':
        ModelClass = cfg['mntp_class']
    elif task_type == 'classification':
        ModelClass = cfg['seq_class']
    else:
        raise ValueError(f"Invalid task_type '{task_type}'.")
    model = ModelClass.from_pretrained(model_path, config=config)

    if peft_weights_path:
        model = PeftModel.from_pretrained(model, peft_weights_path)
    elif peft_config:
        model = get_peft_model(model, peft_config)

    return model, tokenizer


def save_custom_model(
    custom_model_type: str,
    model: Any,
    tokenizer: PreTrainedTokenizer,
    save_dir: str | Path,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    private: bool = True,
) -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = CUSTOM_MODEL_CONFIGS.get(custom_model_type)

    model.save_pretrained(out_dir)
    try:
        print(model.config)
        model.config.save_pretrained(out_dir)
    except AttributeError:
        pass
    tokenizer.save_pretrained(out_dir)

    cfg_key = model.config.__class__.__name__.lower().replace('config', '')
    cfg = CUSTOM_MODEL_CONFIGS.get(cfg_key)
    if cfg:
        src = cfg['modeling_py_path']
        if src.exists():
            (out_dir / src.name).write_bytes(src.read_bytes())

    config_path = out_dir / 'config.json'
    if config_path.exists():
        cfg_json = json.loads(config_path.read_text(encoding='utf-8'))

        def make_value(cls: type) -> str:
            if cls is None:
                return ''
            name = cls.__name__
            if is_transformers_class(cls):
                return ''
            else:
                return f'{repo_id}--{src.stem}.{name}'

        config_cls = cfg.get('config_class') if cfg else model.config.__class__
        base_cls = cfg.get('base_class') if cfg else None
        mntp_cls = cfg.get('mntp_class') if cfg else None
        seq_cls = cfg.get('seq_class') if cfg else None

        desired = {
            'AutoConfig': make_value(config_cls),
            'AutoModel': make_value(base_cls),
            'AutoModelForCausalLM': make_value(mntp_cls),
            'AutoModelForSequenceClassification': make_value(seq_cls),
        }

        existing = cfg_json.get('auto_map', {})
        existing.update({k: v for k, v in desired.items() if v})
        cfg_json['auto_map'] = existing

        config_path.write_text(json.dumps(cfg_json, indent=2, ensure_ascii=False), encoding='utf-8')

    if push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_large_folder(
            folder_path=str(out_dir),
            repo_id=repo_id,
            repo_type='model',
            allow_patterns='*.*',
            private=private,
        )


class CLIConfig(BaseModel):
    model_name_or_path: str | Path = Field(..., description='Model path or HF identifier.')
    custom_model_type: str = Field(
        ...,
        description=f'Custom model family: {", ".join(CUSTOM_MODEL_CONFIGS.keys())}',
    )
    task_type: str = Field('mntp', description=f'Task type: {", ".join(TASK_TYPES)}')
    output_dir: str | Path = Field(..., description='Directory to save files.')
    push_to_hub: bool = Field(False, description='Whether to push to the Hub.')
    repo_id: str | None = Field(None, description='HF repo ID if pushing.')
    private: bool = Field(True, description='Make Hub repo private.')

    @field_validator('custom_model_type')
    def validate_custom_model_type(cls, v: str) -> str:
        if v not in CUSTOM_MODEL_CONFIGS:
            raise ValueError(f"Invalid custom_model_type '{v}'.")
        return v

    @field_validator('task_type')
    def validate_task_type(cls, v: str) -> str:
        if v not in TASK_TYPES:
            raise ValueError(f"Invalid task_type '{v}'.")
        return v

    @field_validator('repo_id')
    def validate_repo_id(cls, v: str | None, info: ValidationInfo) -> str | None:
        if info.data.get('push_to_hub') and not v:
            raise ValueError('repo_id must be set when push_to_hub=True.')
        return v


def main(config_file_path: str | Path | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))

    logger.info('Loading model %s for task %s', cfg.model_name_or_path, cfg.task_type)
    model, tokenizer = load_custom_model(
        cfg.model_name_or_path,
        cfg.custom_model_type,
        cfg.task_type,
    )

    logger.info('Saving model to %s', cfg.output_dir)
    save_custom_model(
        cfg.custom_model_type,
        model,
        tokenizer,
        save_dir=cfg.output_dir,
        push_to_hub=cfg.push_to_hub,
        repo_id=cfg.repo_id,
        private=cfg.private,
    )
    logger.info('Done.')


if __name__ == '__main__':
    fire.Fire(main)
