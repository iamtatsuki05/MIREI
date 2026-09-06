"""Load extractive QA models without silently dropping the pretrained backbone."""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoModelForQuestionAnswering, PreTrainedModel

logger = logging.getLogger(__name__)

HEAD_PREFIX = 'qa_outputs.'


def _backbone_missing_keys(loading_info: dict[str, Any]) -> list[str]:
    return [key for key in loading_info.get('missing_keys', []) if not key.startswith(HEAD_PREFIX)]


def load_question_answering_model(model_name_or_path: str, **kwargs: Any) -> PreTrainedModel:
    """Load ``AutoModelForQuestionAnswering`` and fail fast unless only the QA head is newly initialized.

    In transformers 4.5x the decoder QA heads built on ``GenericForQuestionAnswering`` (Llama, Qwen2, ...) keep
    ``base_model_prefix = "transformer"`` for backward compatibility, while their checkpoints store the backbone
    under ``model.*``. ``from_pretrained`` then reports the whole backbone as "newly initialized" and fine-tuning
    starts from random weights. When that happens the class prefix is switched to ``model`` and the load is retried.
    """
    kwargs.pop('output_loading_info', None)
    model, loading_info = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path, output_loading_info=True, **kwargs
    )
    missing = _backbone_missing_keys(loading_info)
    model_cls = type(model)
    if missing and getattr(model_cls, 'base_model_prefix', None) == 'transformer':
        logger.warning(
            '%s dropped %d backbone tensors (base_model_prefix="transformer"); reloading with base_model_prefix="model"',
            model_cls.__name__,
            len(missing),
        )
        model_cls.base_model_prefix = 'model'
        model, loading_info = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path, output_loading_info=True, **kwargs
        )
        missing = _backbone_missing_keys(loading_info)
    if missing:
        raise RuntimeError(
            f'{model_cls.__name__}: {len(missing)} backbone tensors were not loaded from {model_name_or_path} '
            f'(e.g. {missing[:3]}); refusing to fine-tune a randomly initialized backbone'
        )
    logger.info('qa_model_loaded class=%s newly_initialized=%s', model_cls.__name__, loading_info.get('missing_keys'))
    return model
