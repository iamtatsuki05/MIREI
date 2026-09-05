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
    packing_mask_document_starts: bool = Field(
        True,
        description='When packing, exclude the first token of every document from the causal LM loss.',
    )
    max_seq_length: int | None = Field(
        default=None,
        description={
            'help': (
                'Optional input sequence length after tokenization. '
                'The training dataset will be truncated in block of this size for training. '
                'Default to the model max input length for single sentence inputs (take into account special tokens).'
            )
        },
    )
    overwrite_cache: bool = Field(False, description={'help': 'Overwrite the cached training and evaluation sets'})
    validation_split_percentage: int = Field(
        5,
        description="The percentage of the train set used as validation set in case there's no validation split",
    )
    preprocessing_num_workers: int | None = Field(
        None, description='The number of processes to use for the preprocessing.'
    )
    keep_linebreaks: bool = Field(
        default=True, description={'help': 'Whether to keep line breaks when using TXT files or not.'}
    )

    # Original
    text_column_name: str = Field(
        'text',
        description={
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
