import json

import pytest

from mirei.constract_llm.train.universal_checkpoint import (
    apply_universal_checkpoint_option,
    detect_optimizer_state_keys,
)


def test_apply_is_noop_when_flag_absent_or_false():
    config = {'deepspeed': '/path/ds.json', 'output_dir': '/tmp/out'}
    assert apply_universal_checkpoint_option(config) is config
    config_off = {**config, 'universal_checkpoint_resume': False}
    assert apply_universal_checkpoint_option(config_off) is config_off


def test_apply_injects_load_universal_from_path(tmp_path):
    ds_path = tmp_path / 'ds_zero2.json'
    ds_path.write_text(json.dumps({'zero_optimization': {'stage': 2}, 'bf16': {'enabled': True}}))
    config = {'universal_checkpoint_resume': True, 'deepspeed': str(ds_path)}

    updated = apply_universal_checkpoint_option(config)

    assert updated['deepspeed']['checkpoint'] == {'load_universal': True}
    assert updated['deepspeed']['zero_optimization'] == {'stage': 2}
    # 元の config dict と ds_config ファイルは書き換えない
    assert config['deepspeed'] == str(ds_path)
    assert 'checkpoint' not in json.loads(ds_path.read_text())


def test_apply_injects_load_universal_into_dict_without_mutation():
    ds_config = {'zero_optimization': {'stage': 2}, 'checkpoint': {'tag_validation': 'Warn'}}
    config = {'universal_checkpoint_resume': True, 'deepspeed': ds_config}

    updated = apply_universal_checkpoint_option(config)

    assert updated['deepspeed']['checkpoint'] == {'tag_validation': 'Warn', 'load_universal': True}
    assert ds_config['checkpoint'] == {'tag_validation': 'Warn'}


def test_apply_requires_deepspeed_config():
    with pytest.raises(ValueError, match='requires a "deepspeed" config'):
        apply_universal_checkpoint_option({'universal_checkpoint_resume': True})


def test_apply_rejects_unsupported_deepspeed_type():
    with pytest.raises(TypeError):
        apply_universal_checkpoint_option({'universal_checkpoint_resume': True, 'deepspeed': 123})


@pytest.mark.parametrize(
    ('state_group', 'expected'),
    [
        ({'exp_avg': object(), 'exp_avg_sq': object()}, ['exp_avg', 'exp_avg_sq']),
        ({'z': object(), 'exp_avg_sq': object()}, ['exp_avg_sq', 'z']),
        ({'z': object(), 'exp_avg_sq': object(), 'step': 100}, ['exp_avg_sq', 'z']),
    ],
)
def test_detect_optimizer_state_keys(state_group, expected):
    assert detect_optimizer_state_keys(state_group) == expected


def test_detect_optimizer_state_keys_rejects_empty():
    with pytest.raises(ValueError):
        detect_optimizer_state_keys({'step': 100})
