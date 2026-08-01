import inspect
import json
from pathlib import Path
from typing import Any, Final

import transformers
from peft import PeftConfig, PeftModel, get_peft_model
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    MistralConfig,
    PreTrainedTokenizer,
    Qwen2Config,
)

from mirei.constract_llm.model.custom.modeling_bidirectional_llama import (
    LlamaBiForMNTP,
    LlamaBiForSequenceClassification,
    LlamaBiModel,
)
from mirei.constract_llm.model.custom.modeling_bidirectional_mistral import (
    MistralBiForMNTP,
    MistralBiForSequenceClassification,
    MistralBiModel,
)
from mirei.constract_llm.model.custom.modeling_bidirectional_qwen2 import (
    Qwen2BiForMNTP,
    Qwen2BiForSequenceClassification,
    Qwen2BiModel,
)
from mirei.env import PACKAGE_DIR

CUSTOM_MODEL_CONFIGS: Final[dict[str, dict[str, Any]]] = {
    'llama': {
        'config_class': LlamaConfig,
        'base_class': LlamaBiModel,
        'mntp_class': LlamaBiForMNTP,
        'seq_class': LlamaBiForSequenceClassification,
        'modeling_py_path': PACKAGE_DIR / 'src/mirei/constract_llm/model/custom/modeling_bidirectional_llama.py',
    },
    'mistral': {
        'config_class': MistralConfig,
        'base_class': MistralBiModel,
        'mntp_class': MistralBiForMNTP,
        'seq_class': MistralBiForSequenceClassification,
        'modeling_py_path': PACKAGE_DIR / 'src/mirei/constract_llm/model/custom/modeling_bidirectional_mistral.py',
    },
    'qwen2': {
        'config_class': Qwen2Config,
        'base_class': Qwen2BiModel,
        'mntp_class': Qwen2BiForMNTP,
        'seq_class': Qwen2BiForSequenceClassification,
        'modeling_py_path': PACKAGE_DIR / 'src/mirei/constract_llm/model/custom/modeling_bidirectional_qwen2.py',
    },
}

TASK_TYPES: Final[list[str]] = ['lm', 'mntp', 'classification']


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
    revision: str | None = 'main',
) -> tuple[Any, PreTrainedTokenizer]:
    if custom_model_type not in CUSTOM_MODEL_CONFIGS:
        raise ValueError(f"Invalid custom_model_type '{custom_model_type}'.")

    set_auto_model_classes(custom_model_type)

    cfg = CUSTOM_MODEL_CONFIGS[custom_model_type]
    model_source = str(model_name_or_path)
    revision_kwargs = {'revision': revision} if revision is not None else {}

    config = cfg['config_class'].from_pretrained(model_source, **revision_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_source, **revision_kwargs)

    if task_type == 'lm':
        ModelClass = cfg['base_class']
    elif task_type == 'mntp':
        ModelClass = cfg['mntp_class']
    elif task_type == 'classification':
        ModelClass = cfg['seq_class']
    else:
        raise ValueError(f"Invalid task_type '{task_type}'.")
    model = ModelClass.from_pretrained(model_source, config=config, **revision_kwargs)

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
    if cfg is None:
        raise ValueError(f"No custom model export configuration for config key '{cfg_key}'.")
    src = cfg['modeling_py_path']
    if not src.is_file():
        raise FileNotFoundError(f'Missing custom model Python artifact: {src}')
    destination = out_dir / src.name
    destination.unlink(missing_ok=True)
    destination.write_bytes(src.read_bytes())

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

        config_cls = cfg['config_class']
        base_cls = cfg['base_class']
        mntp_cls = cfg['mntp_class']
        seq_cls = cfg['seq_class']

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
