from types import SimpleNamespace

import pytest
import torch
from sentence_transformers import SentenceTransformer, models
from torch import nn

from mirei.constract_llm.train.st.pooling import check_special_tokens, replace_pooling

BOS, EOS = 1, 2


class _Stub(nn.Module):
    def __init__(self, ids: list[int] | None = None) -> None:
        super().__init__()
        self.ids = ids or [5]
        self.tokenizer = SimpleNamespace(bos_token_id=BOS, eos_token_id=EOS)

    def forward(self, features: dict) -> dict:
        return features

    def tokenize(self, texts: list[str], **kwargs: object) -> dict:
        return {'input_ids': torch.tensor([self.ids])}


@pytest.mark.parametrize(
    ('ids', 'add_bos', 'add_eos'),
    [([BOS, 5], True, None), ([5, EOS], None, True), ([5], None, None), ([5], False, False)],
)
def test_check_special_tokens_accepts_applied_settings(
    ids: list[int], add_bos: bool | None, add_eos: bool | None
) -> None:
    model = SentenceTransformer(modules=[_Stub(ids), models.Pooling(16)])

    check_special_tokens(model, add_bos, add_eos)


@pytest.mark.parametrize(('add_bos', 'add_eos', 'match'), [(True, None, 'BOS'), (None, True, 'EOS')])
def test_check_special_tokens_rejects_ignored_settings(add_bos: bool | None, add_eos: bool | None, match: str) -> None:
    model = SentenceTransformer(modules=[_Stub([5]), models.Pooling(16)])

    with pytest.raises(ValueError, match=match):
        check_special_tokens(model, add_bos, add_eos)


def test_replace_pooling_swaps_mode_and_keeps_dimension() -> None:
    model = SentenceTransformer(modules=[_Stub(), models.Pooling(16, pooling_mode='mean')])

    replace_pooling(model, 'lasttoken')

    assert isinstance(model[1], models.Pooling)
    assert model[1].get_pooling_mode_str() == 'lasttoken'
    assert model[1].get_sentence_embedding_dimension() == 16


def test_replace_pooling_rejects_unexpected_module_stack() -> None:
    model = SentenceTransformer(modules=[_Stub(), models.Pooling(16), models.Normalize()])

    with pytest.raises(ValueError, match='Transformer, Pooling'):
        replace_pooling(model, 'cls')
