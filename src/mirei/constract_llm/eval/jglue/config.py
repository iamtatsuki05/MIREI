from typing import Final

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
