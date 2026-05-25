import pytest

from mirei.constract_llm.eval.jglue.config import get_benchmark_dataset_spec


def test_jglue_task_uses_jglue_dataset() -> None:
    spec = get_benchmark_dataset_spec('JNLI')

    assert spec.dataset_name == 'shunk031/JGLUE'
    assert spec.dataset_config_name == 'JNLI'
    assert spec.benchmark_name == 'jglue'


def test_glue_task_uses_glue_dataset() -> None:
    spec = get_benchmark_dataset_spec('mnli')

    assert spec.dataset_name == 'nyu-mll/glue'
    assert spec.dataset_config_name == 'mnli'
    assert spec.benchmark_name == 'glue'


def test_task_dataset_can_be_overridden() -> None:
    spec = get_benchmark_dataset_spec('sst2', dataset_name='glue', dataset_config_name='sst2')

    assert spec.dataset_name == 'glue'
    assert spec.dataset_config_name == 'sst2'
    assert spec.benchmark_name == 'glue'


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_benchmark_dataset_spec('unknown')
