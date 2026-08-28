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
    # Original
    text_column_name: str = Field(
        'text',
        metadata={
            'help': (
                "The name of the column containing the text to tokenize (for datasets that don't have a 'text' column)."
            )
        },
    )

    universal_checkpoint_resume: bool = Field(
        False,
        description=(
            'One-shot resume from a DeepSpeed Universal Checkpoint (converted with '
            'scripts/constract_llm/train/tools/convert_zero_checkpoint_to_universal.py) so the run '
            'can continue on a different number of GPUs. Injects checkpoint.load_universal into the '
            'DeepSpeed config; adjust gradient_accumulation_steps inversely to keep the global batch, '
            'and disable this flag again after the first successful resume.'
        ),
    )

    def __post_init__(self):
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
