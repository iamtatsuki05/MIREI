from pydantic import Field
from pydantic.dataclasses import dataclass
from transformers.utils.versions import require_version


@dataclass
class DataTrainingArguments:
    dataset_name: str | None = Field(None, description='The name of the dataset to use (via the datasets library).')
    dataset_config_name: str | None = Field(
        None,
        description='The configuration name of the dataset to use (via the datasets library).',
    )
    tokenized_dataset_path: str | None = Field(
        None,
        description='Path to a preprocessed DatasetDict saved with datasets.save_to_disk.',
    )
    tokenized_train_files: str | None = Field(
        None,
        description='Glob pattern or file path for preprocessed tokenized train Arrow shard(s).',
    )
    tokenized_validation_files: str | None = Field(
        None,
        description='Glob pattern or file path for preprocessed tokenized validation Arrow shard(s).',
    )
    tokenized_test_files: str | None = Field(
        None,
        description='Glob pattern or file path for preprocessed tokenized test Arrow shard(s).',
    )
    train_file: str | None = Field(None, description='The input training data file (a text file).')
    validation_file: str | None = Field(
        None,
        description='An optional input evaluation data file to evaluate the perplexity on (a text file).',
    )
    overwrite_cache: bool = Field(False, metadata={'help': 'Overwrite the cached training and evaluation sets'})
    validation_split_percentage: int = Field(
        5,
        description="The percentage of the train set used as validation set in case there's no validation split",
    )
    max_seq_length: int | None = Field(
        None,
        description=(
            'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.'
        ),
    )
    preprocessing_num_workers: int | None = Field(
        None, description='The number of processes to use for the preprocessing.'
    )
    mlm_probability: float = Field(0.15, description='Ratio of tokens to mask for masked language modeling loss')
    line_by_line: bool = Field(
        False,
        description='Whether distinct lines of text in the dataset are to be handled as distinct sequences.',
    )
    pad_to_max_length: bool = Field(
        False,
        description=(
            'Whether to pad all samples to `max_seq_length`. '
            'If False, will pad the samples dynamically when batching to the maximum length in the batch.'
        ),
    )
    max_train_samples: int | None = Field(
        None,
        description=(
            'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'
        ),
    )
    max_eval_samples: int | None = Field(
        None,
        description=(
            'For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.'
        ),
    )
    streaming: bool = Field(False, description='Enable streaming mode')
    packing: bool = Field(
        False,
        description=(
            'Pack multiple documents into one row of `packing_seq_length` tokens without letting attention cross '
            'document boundaries (see mirei.constract_llm.train.language_model.packing).'
        ),
    )
    packing_strategy: str = Field(
        'bfd',
        description=(
            "'bfd' packs whole documents (best-fit decreasing); 'wrapped' concatenates and chunks like group_texts "
            '(documents are split, no boundary information).'
        ),
    )
    packing_seq_length: int | None = Field(
        None, description='Row length used for packing. Defaults to `max_seq_length`.'
    )
    packing_encoder_mode: str = Field(
        'auto',
        description=(
            "How document boundaries are passed to the encoder when packing: 'unpad' (rows are split back into "
            "documents; ModernBERT re-packs them with cu_seqlens), 'mask' (3D block-diagonal attention mask + "
            "per-document position_ids for BERT-style encoders) or 'auto' (unpad for ModernBERT, mask otherwise)."
        ),
    )
    # Original
    text_column_name: str = Field(
        'text',
        metadata={
            'help': (
                "The name of the column containing the text to tokenize (for datasets that don't have a 'text' column)."
            )
        },
    )

    def __post_init__(self):
        if self.packing_strategy not in ('bfd', 'wrapped'):
            raise ValueError(f"packing_strategy must be 'bfd' or 'wrapped', got {self.packing_strategy!r}")
        if self.packing_seq_length is not None and self.packing_seq_length <= 0:
            raise ValueError('packing_seq_length must be a positive integer')
        if self.packing_encoder_mode not in ('auto', 'unpad', 'mask'):
            raise ValueError(
                f"packing_encoder_mode must be 'auto', 'unpad' or 'mask', got {self.packing_encoder_mode!r}"
            )
        if self.packing and not self.line_by_line:
            raise ValueError('packing requires line_by_line=True (documents must be tokenized individually)')
        if self.streaming:
            require_version('datasets>=2.0.0', 'The streaming feature requires `datasets>=2.0.0`')
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.tokenized_dataset_path is None
            and self.tokenized_train_files is None
        ):
            raise ValueError('Need either a dataset name or a training/validation file.')
        else:
            if self.train_file is not None:
                extension = self.train_file.split('.')[-1]
                if extension not in ['csv', 'json', 'txt']:
                    raise ValueError('`train_file` should be a csv, a json or a txt file.')
            if self.validation_file is not None:
                extension = self.validation_file.split('.')[-1]
                if extension not in ['csv', 'json', 'txt']:
                    raise ValueError('`validation_file` should be a csv, a json or a txt file.')
