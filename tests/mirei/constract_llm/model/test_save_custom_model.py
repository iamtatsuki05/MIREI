import json

import pytest
from transformers import LlamaConfig

from mirei.constract_llm.model.custom.modeling_bidirectional_llama import LlamaBiModel
from mirei.constract_llm.model.save_custom_model import save_custom_model


class _Tokenizer:
    def save_pretrained(self, path):
        pass


@pytest.fixture(scope='module')
def tiny_bi_llama():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    return LlamaBiModel(config)


def _auto_map(path):
    return json.loads((path / 'config.json').read_text(encoding='utf-8'))['auto_map']


def test_local_save_points_auto_map_at_the_copied_modeling_file(tmp_path, tiny_bi_llama):
    save_custom_model('llama', tiny_bi_llama, _Tokenizer(), tmp_path)

    assert _auto_map(tmp_path) == {
        'AutoModel': 'modeling_bidirectional_llama.LlamaBiModel',
        'AutoModelForCausalLM': 'modeling_bidirectional_llama.LlamaBiForMNTP',
        'AutoModelForSequenceClassification': 'modeling_bidirectional_llama.LlamaBiForSequenceClassification',
    }
    assert (tmp_path / 'modeling_bidirectional_llama.py').is_file()


def test_hub_save_prefixes_auto_map_with_the_repo_id(tmp_path, tiny_bi_llama):
    save_custom_model('llama', tiny_bi_llama, _Tokenizer(), tmp_path, repo_id='user/model')

    assert _auto_map(tmp_path)['AutoModel'] == 'user/model--modeling_bidirectional_llama.LlamaBiModel'
