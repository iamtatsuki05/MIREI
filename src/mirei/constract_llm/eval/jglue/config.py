from typing import Final, NamedTuple

TASK_TO_KEYS: Final[dict[str, tuple[str | None, str | None]]] = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
    'JSTS': ('sentence1', 'sentence2'),
    'JNLI': ('sentence1', 'sentence2'),
    'JCoLA': ('sentence', None),
}

GLUE_TASKS: Final[frozenset[str]] = frozenset(
    {
        'cola',
        'mnli',
        'mrpc',
        'qnli',
        'qqp',
        'rte',
        'sst2',
        'stsb',
        'wnli',
    }
)
JGLUE_TASKS: Final[frozenset[str]] = frozenset({'JSTS', 'JNLI', 'JCoLA'})


class BenchmarkDatasetSpec(NamedTuple):
    dataset_name: str
    dataset_config_name: str
    benchmark_name: str


def get_benchmark_dataset_spec(
    task_name: str,
    dataset_name: str | None = None,
    dataset_config_name: str | None = None,
) -> BenchmarkDatasetSpec:
    if task_name in JGLUE_TASKS:
        return BenchmarkDatasetSpec(
            dataset_name=dataset_name or 'shunk031/JGLUE',
            dataset_config_name=dataset_config_name or task_name,
            benchmark_name='jglue',
        )
    if task_name in GLUE_TASKS:
        return BenchmarkDatasetSpec(
            dataset_name=dataset_name or 'nyu-mll/glue',
            dataset_config_name=dataset_config_name or task_name,
            benchmark_name='glue',
        )
    raise ValueError('Unknown task, you should pick one in ' + ','.join(TASK_TO_KEYS.keys()))
