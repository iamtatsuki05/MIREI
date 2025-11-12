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
        if self.streaming:
            require_version('datasets>=2.0.0', 'The streaming feature requires `datasets>=2.0.0`')
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
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
