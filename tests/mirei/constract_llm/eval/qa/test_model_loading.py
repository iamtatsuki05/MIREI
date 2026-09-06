import pytest
import torch
from transformers import AutoModelForQuestionAnswering, LlamaConfig, LlamaForCausalLM

from mirei.constract_llm.eval.qa.model_loading import _backbone_missing_keys, load_question_answering_model


@pytest.fixture(scope='module')
def tiny_llama_dir(tmp_path_factory):
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    path = tmp_path_factory.mktemp('tiny-llama')
    model.save_pretrained(path)
    return path, model


def test_backbone_weights_are_loaded(tiny_llama_dir):
    path, causal = tiny_llama_dir
    cls = AutoModelForQuestionAnswering._model_mapping[LlamaConfig]
    original_prefix = cls.base_model_prefix
    try:
        model = load_question_answering_model(str(path), torch_dtype=torch.float32)
        backbone = getattr(model, type(model).base_model_prefix)
        assert torch.equal(backbone.embed_tokens.weight, causal.model.embed_tokens.weight)
        assert torch.equal(backbone.layers[0].self_attn.q_proj.weight, causal.model.layers[0].self_attn.q_proj.weight)
    finally:
        cls.base_model_prefix = original_prefix


def test_plain_auto_class_reports_missing_backbone_when_prefix_is_transformer(tiny_llama_dir):
    path, _ = tiny_llama_dir
    _, info = AutoModelForQuestionAnswering.from_pretrained(str(path), output_loading_info=True)
    cls = AutoModelForQuestionAnswering._model_mapping[LlamaConfig]
    if cls.base_model_prefix != 'transformer':
        pytest.skip('this transformers version loads the backbone with the default prefix')
    assert any(not k.startswith('qa_outputs.') for k in info['missing_keys'])


def test_task_head_keys_are_not_backbone_keys():
    info = {'missing_keys': ['classifier.weight', 'classifier.bias', 'head.dense.weight', 'qa_outputs.weight']}
    assert _backbone_missing_keys(info, 'model') == []
    info = {'missing_keys': ['transformer.embed_tokens.weight', 'qa_outputs.weight']}
    assert _backbone_missing_keys(info, 'transformer') == ['transformer.embed_tokens.weight']


def test_encoder_with_extra_head_parameters_loads(tmp_path):
    modernbert = pytest.importorskip('transformers.models.modernbert.modeling_modernbert')
    config = modernbert.ModernBertConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        reference_compile=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sep_token_id=2,
        cls_token_id=1,
    )
    modernbert.ModernBertForMaskedLM(config).save_pretrained(tmp_path)
    model = load_question_answering_model(str(tmp_path), torch_dtype=torch.float32)
    assert type(model).__name__ == 'ModernBertForQuestionAnswering'
