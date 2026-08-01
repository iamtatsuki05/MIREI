import torch
from transformers import LlamaConfig

from mirei.constract_llm.eval.analysis.metrics import bidirectional_forward_mask, build_model_inputs
from mirei.constract_llm.model.custom.modeling_bidirectional_llama import LlamaBiModel


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attn_implementation='eager',
    )


def _future_mass(attentions: tuple[torch.Tensor, ...], mask: torch.Tensor) -> float:
    total = 0.0
    seq_len = mask.size(1)
    future = torch.arange(seq_len).view(1, -1) > torch.arange(seq_len).view(-1, 1)
    for layer_attention in attentions:
        total += float((layer_attention[0] * future).sum())
    return total


def test_bidirectional_forward_mask_shape_and_values() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    mask = bidirectional_forward_mask(attention_mask, torch.float32)
    assert mask.shape == (2, 1, 1, 4)
    assert torch.all(mask[0, 0, 0, :3] == 0)
    assert mask[0, 0, 0, 3] == torch.finfo(torch.float32).min
    assert torch.all(mask[1, 0, 0, 2:] == torch.finfo(torch.float32).min)


def test_build_model_inputs_replaces_mask_only_when_needed() -> None:
    encoded = {'input_ids': torch.ones(1, 4, dtype=torch.long), 'attention_mask': torch.ones(1, 4, dtype=torch.long)}
    unchanged = build_model_inputs(encoded, needs_bidirectional_mask=False, model_dtype=torch.float32)
    assert unchanged['attention_mask'].dim() == 2
    replaced = build_model_inputs(encoded, needs_bidirectional_mask=True, model_dtype=torch.float32)
    assert replaced['attention_mask'].shape == (1, 1, 1, 4)
    assert encoded['attention_mask'].dim() == 2


def test_llama_bi_model_needs_explicit_4d_mask_under_eager() -> None:
    """Regression test for the transformers-4.56 eager gap that motivated the 4D mask.

    LlamaBiModel's bidirectional hooks only take effect on the flash-attention path, so
    under eager a plain 2D mask still yields strictly causal attention; the explicit 4D
    padding-only mask restores attention to future tokens.
    """
    torch.manual_seed(0)
    model = LlamaBiModel(_tiny_config())
    model.eval()
    input_ids = torch.randint(0, 128, (1, 8))
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    with torch.inference_mode():
        causal_out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        inputs = build_model_inputs(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            needs_bidirectional_mask=True,
            model_dtype=torch.float32,
        )
        bi_out = model(**inputs, output_attentions=True)
    assert _future_mass(causal_out.attentions, attention_mask) == 0.0
    assert _future_mass(bi_out.attentions, attention_mask) > 0.0
