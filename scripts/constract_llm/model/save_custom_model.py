import logging
from pathlib import Path
from typing import Any

import fire
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from mirei.common.utils.cli_utils import load_cli_config
from mirei.constract_llm.model.save_custom_model import (
    CUSTOM_MODEL_CONFIGS,
    TASK_TYPES,
    load_custom_model,
    save_custom_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
