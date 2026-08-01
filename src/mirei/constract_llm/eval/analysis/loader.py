import logging
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_backbone(
    model_name_or_path: str,
    revision: str | None,
    device: str,
    torch_dtype: torch.dtype,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, bool]:
    """
    Load a transformer backbone and tokenizer for geometry / attention analysis.

    If `model_name_or_path` is a local path containing `modules.json`, it is loaded as a
    SentenceTransformer and the backbone (`auto_model`) and tokenizer of the first module
    are extracted. Otherwise it is loaded with `AutoModel` / `AutoTokenizer`
    (`trust_remote_code=True`). In both cases `attn_implementation='eager'` is forced so
    that attention weights can be extracted, and the model is set to eval mode.
    """
    modules_json_path = Path(model_name_or_path) / 'modules.json'
    if modules_json_path.is_file():
        logger.info(f'Loading {model_name_or_path} as SentenceTransformer (modules.json found)')
        st_kwargs: dict[str, Any] = {
            'device': device,
            'trust_remote_code': True,
            'model_kwargs': {'torch_dtype': torch_dtype, 'attn_implementation': 'eager'},
        }
        if revision is not None:
            st_kwargs['revision'] = revision
        st_model = SentenceTransformer(model_name_or_path, **st_kwargs)
        first_module = st_model._first_module()
        auto_model = first_module.auto_model
        tokenizer = first_module.tokenizer
    else:
        logger.info(f'Loading {model_name_or_path} with AutoModel/AutoTokenizer')
        auto_model = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation='eager',
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision, trust_remote_code=True)
        auto_model.to(device)
    reported_backend = getattr(auto_model.config, '_attn_implementation', None)
    if reported_backend != 'eager':
        raise RuntimeError(f"requested attention backend 'eager', but model reported {reported_backend!r}")
    logger.info(f'Loaded backbone {type(auto_model).__name__} (attn_implementation={reported_backend!r})')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    auto_model.eval()
    # LlamaBiModel (covers the Llama-family Bi decoders incl. sarashina2.2-Bi) only disables
    # causality via hooks that take effect on the flash-attention path of transformers 4.56;
    # under eager the inherited LlamaModel.forward rebuilds a causal mask, so callers must pass
    # the explicit 4D mask from `metrics.bidirectional_forward_mask`. The Qwen2/Mistral Bi
    # variants override `forward` itself and are bidirectional on every backend, so they are
    # deliberately excluded here (the custom modeling code itself is supported on FA2 only).
    class_name = type(auto_model).__name__
    needs_bidirectional_mask = class_name == 'LlamaBiModel'
    if class_name.endswith('BiModel') and not needs_bidirectional_mask:
        logger.info(f'{class_name} builds its own bidirectional mask; no 4D mask override applied')
    logger.info(f'needs_bidirectional_mask={needs_bidirectional_mask}')
    return auto_model, tokenizer, needs_bidirectional_mask
