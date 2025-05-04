from pydantic import Field
from pydantic.dataclasses import dataclass
from transformers import MODEL_FOR_MASKED_LM_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str | None = Field(
        None,
        description=(
            "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        ),
    )
    model_type: str | None = Field(
        None,
        description='If training from scratch, pass a model type from the list: ' + ', '.join(MODEL_TYPES),
    )
    config_overrides: str | None = Field(
        None,
        description=(
            'Override some existing default config settings when training from scratch. '
            'Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index'
        ),
    )
    config_name: str | None = Field(None, description='Pretrained config name or path if not the same as model_name')
    tokenizer_name: str | None = Field(
        None,
        description='Pretrained tokenizer name or path if not the same as model_name',
    )
    cache_dir: str | None = Field(
        None,
        description='Where to store the pretrained models downloaded from huggingface.co',
    )
    use_fast_tokenizer: bool = Field(
        True,
        description='Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.',
    )
    model_revision: str = Field(
        'main',
        description='The specific model version to use (can be a branch name, tag name or commit id).',
    )
    token: str | None = Field(
        None,
        description=(
            'The token to use as HTTP bearer authorization for remote files. '
            'If not specified, will use the token generated when running `huggingface-cli login`.'
        ),
    )
    trust_remote_code: bool = Field(
        False,
        description=(
            'Whether to trust the execution of code from datasets/models defined on the Hub. '
            'This option should only be set to `True` for repositories you trust.'
        ),
    )
    torch_dtype: str | None = Field(
        None,
        description=(
            'Override the default `torch.dtype` and load the model under this dtype. '
            "If `auto` is passed, the dtype will be automatically derived from the model's weights."
        ),
    )
    low_cpu_mem_usage: bool = Field(
        False,
        description=(
            'Create the model as an empty shell and only materialize its parameters '
            'when the pretrained weights are loaded. Reduces peak CPU RAM usage.'
        ),
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
