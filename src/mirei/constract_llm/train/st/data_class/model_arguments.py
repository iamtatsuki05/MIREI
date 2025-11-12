from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class ModelArguments:
    model_name_or_path: str = Field(
        default=None,
        metadata={
            'help': 'The model checkpoint for weights initialization. '
            "Don't set if you want to train a model from scratch."
        },
    )
    cache_dir: str | None = Field(
        default=None,
        metadata={'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'},
    )
    use_fast_tokenizer: bool = Field(
        True,
        description='Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.',
    )
    model_revision: str = Field(
        default='main',
        metadata={'help': 'The specific model version to use (can be a branch name, tag name or commit id).'},
    )
    token: str | None = Field(
        default=None,
        metadata={
            'help': (
                'The token to use as HTTP bearer authorization for remote files. If not specified, will use the token '
                'generated when running `huggingface-cli login` (stored in `~/.huggingface`).'
            )
        },
    )
    use_auth_token: bool | None = Field(
        default=None,
        metadata={
            'help': 'The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.'
        },
    )
    trust_remote_code: bool = Field(
        default=False,
        metadata={
            'help': (
                'Whether or not to allow for custom models defined on the Hub in their own modeling files. This option '
                'should only be set to `True` for repositories you trust and in which you have read the code, as it will '
                'execute code present on the Hub on your local machine.'
            )
        },
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
    attn_implementation: str = Field(
        default='sdpa',
        metadata={
            'help': ('The attention implementation to use in the model.'),
            'choices': ['eager', 'sdpa', 'flash_attention_2'],
        },
    )
    # Optional
    loss_cache_mini_batch_size: int | None = Field(
        default=None,
        metadata={'help': 'The mini batch size for cached loss.'},
    )
    loss_scale: float = Field(
        default=20.0,
        metadata={
            'help': (
                'The scale factor for the loss function. '
                'This is used to adjust the magnitude of the loss during training.'
            )
        },
    )
