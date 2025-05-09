from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class CustomArguments:
    lora: bool = Field(default=False, metadata={'help': 'Whether to use lora or not'})

    lora_dropout: float = Field(default=0.05, metadata={'help': 'The dropout rate for lora'})

    lora_r: int = Field(default=8, metadata={'help': 'The r value for lora'})

    mask_token_type: str = Field(
        default='blank',
        metadata={'help': 'The type of mask token. Options: blank, eos, mask'},
    )

    stop_after_n_steps: int = Field(default=100000000, metadata={'help': 'Stop training after n steps'})

    data_collator_type: str = Field(
        default='default',
        metadata={'help': 'The type of data collator. Options: default, all_mask'},
    )
