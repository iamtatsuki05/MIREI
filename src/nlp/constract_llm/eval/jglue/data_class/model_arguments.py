from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = Field(
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    config_name: str | None = Field(
        default=None,
        metadata={'help': 'Pretrained config name or path if not the same as model_name'},
    )
    tokenizer_name: str | None = Field(
        default=None,
        metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'},
    )
    cache_dir: str | None = Field(
        default=None,
        metadata={'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'},
    )
    use_fast_tokenizer: bool = Field(
        default=True,
        metadata={'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'},
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
    trust_remote_code: bool = Field(
        default=False,
        metadata={
            'help': (
                'Whether to trust the execution of code from datasets/models defined on the Hub.'
                ' This option should only be set to `True` for repositories you trust and in which you have read the'
                ' code, as it will execute code present on the Hub on your local machine.'
            )
        },
    )
    ignore_mismatched_sizes: bool = Field(
        default=False,
        metadata={'help': 'Will enable to load a pretrained model whose head dimensions are different.'},
    )
