import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertForMaskedLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from mirei.constract_llm.train.language_model.clm.data_class.data_training_arguments import (
    DataTrainingArguments as ClmDataTrainingArguments,
)
from mirei.constract_llm.train.language_model.mlm.data_class.data_training_arguments import (
    DataTrainingArguments as MlmDataTrainingArguments,
)
from mirei.constract_llm.train.language_model.packing import (
    PackedCausalLMCollator,
    PackedMaskedLMCollator,
    block_diagonal_attention_mask,
    flash_attention_varlen_kwargs,
    pack_tokenized_dataset,
    segment_ids,
    segment_position_ids,
    select_encoder_packing_mode,
)

VOCAB = 64
PAD, CLS, SEP, MASK = 0, 1, 2, 3


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f'[{name}]': i for i, name in enumerate(['PAD', 'CLS', 'SEP', 'MASK'])}
    vocab.update({f'w{i}': i for i in range(4, VOCAB)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token='[PAD]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]'
    )


def _docs(lengths: list[int], with_special: bool = False) -> list[list[int]]:
    g = torch.Generator().manual_seed(0)
    docs = []
    for n in lengths:
        body = torch.randint(4, VOCAB, (n - 2 if with_special else n,), generator=g).tolist()
        docs.append([CLS, *body, SEP] if with_special else body)
    return docs


def _dataset(docs: list[list[int]], with_special_mask: bool = False, with_scalar: bool = False) -> Dataset:
    cols: dict[str, list] = {'input_ids': docs, 'attention_mask': [[1] * len(d) for d in docs]}
    if with_special_mask:
        cols['special_tokens_mask'] = [[1 if t in (CLS, SEP) else 0 for t in d] for d in docs]
    if with_scalar:
        cols['title'] = [f'doc{i}' for i in range(len(docs))]
    return Dataset.from_dict(cols)


def _llama(attn_implementation: str) -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=PAD,
        attn_implementation=attn_implementation,
    )
    return LlamaForCausalLM(config).eval()


def _bert(attn_implementation: str) -> BertForMaskedLM:
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        pad_token_id=PAD,
        attn_implementation=attn_implementation,
    )
    model = BertForMaskedLM(config).eval()
    assert model.config._attn_implementation == attn_implementation
    return model


# ---------------------------------------------------------------- packing


def test_pack_bfd_keeps_documents_whole_and_truncates_long_ones():
    docs = _docs([5, 3, 4, 10, 2])
    packed = pack_tokenized_dataset(_dataset(docs), seq_length=8, strategy='bfd', num_proc=None)
    seen = []
    for row in packed:
        assert len(row['input_ids']) <= 8
        assert sum(row['seq_lengths']) == len(row['input_ids']) == len(row['attention_mask'])
        start = 0
        for n in row['seq_lengths']:
            seen.append(row['input_ids'][start : start + n])
            start += n
    expected = [d[:8] for d in docs]  # the 10-token document is truncated, nothing else changes
    assert sorted(seen) == sorted(expected)


def test_pack_drops_scalar_columns():
    packed = pack_tokenized_dataset(_dataset(_docs([5, 3]), with_scalar=True), seq_length=8, strategy='bfd')
    assert 'title' not in packed.column_names
    assert set(packed.column_names) == {'input_ids', 'attention_mask', 'seq_lengths'}


def test_pack_wrapped_has_no_boundaries():
    docs = _docs([5, 3, 4])
    packed = pack_tokenized_dataset(_dataset(docs), seq_length=4, strategy='wrapped')
    assert 'seq_lengths' not in packed.column_names
    assert all(len(row['input_ids']) <= 4 for row in packed)
    # documents are concatenated and chunked (split across rows), nothing is truncated away
    assert [t for row in packed for t in row['input_ids']] == [t for d in docs for t in d]


@pytest.mark.parametrize('kwargs', [{'strategy': 'random'}, {'seq_length': 0}])
def test_pack_rejects_bad_arguments(kwargs):
    args = {'seq_length': 8, 'strategy': 'bfd'} | kwargs
    with pytest.raises(ValueError):
        pack_tokenized_dataset(_dataset(_docs([3])), **args)


def test_segment_helpers():
    assert segment_position_ids([3, 2]).tolist() == [0, 1, 2, 0, 1]
    assert segment_ids([3, 2]).tolist() == [0, 0, 0, 1, 1]
    mask = block_diagonal_attention_mask(segment_ids([2, 1]), torch.tensor([True, True, False]))
    assert mask.tolist() == [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
    fa = flash_attention_varlen_kwargs([[3, 2], [4, 1]])
    assert fa['cu_seq_lens_q'].tolist() == [0, 3, 5, 9, 10] and fa['cu_seq_lens_q'].dtype == torch.int32
    assert fa['cu_seq_lens_k'].tolist() == [0, 3, 5, 9, 10]
    assert fa['max_length_q'] == fa['max_length_k'] == 4 and isinstance(fa['max_length_q'], int)


# ---------------------------------------------------------------- causal LM


def test_causal_collator_positions_labels_padding_and_varlen_kwargs():
    collator = PackedCausalLMCollator(pad_token_id=PAD)
    batch = collator(
        [
            {'input_ids': [11, 12, 13, 14, 15, 16, 17], 'seq_lengths': [5, 2]},
            {'input_ids': [21, 22, 23], 'seq_lengths': [3]},
        ]
    )
    assert 'attention_mask' not in batch  # sdpa / eager must derive the mask from position_ids
    assert batch['input_ids'].tolist() == [[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, PAD, PAD, PAD, PAD]]
    assert batch['position_ids'].tolist() == [[0, 1, 2, 3, 4, 0, 1], [0, 1, 2, 0, 1, 2, 3]]
    assert batch['labels'].tolist() == [[-100, 12, 13, 14, 15, -100, 17], [-100, 22, 23, -100, -100, -100, -100]]
    # varlen kwargs cover the row-major flattened batch: [5, 2] + [3, 4 (padding segment)]
    assert batch['cu_seq_lens_q'].tolist() == [0, 5, 7, 10, 14]
    assert batch['max_length_q'] == 5

    plain = PackedCausalLMCollator(pad_token_id=PAD, mask_document_starts=False, return_flash_attn_kwargs=False)(
        [{'input_ids': [11, 12, 13], 'seq_lengths': [2, 1]}]
    )
    assert plain['labels'].tolist() == [[11, 12, 13]] and 'cu_seq_lens_q' not in plain


def test_causal_collator_treats_rows_without_seq_lengths_as_one_segment():
    batch = PackedCausalLMCollator(pad_token_id=PAD)([{'input_ids': [11, 12, 13]}])
    assert batch['position_ids'].tolist() == [[0, 1, 2]]
    assert batch['labels'].tolist() == [[11, 12, 13]]  # wrapped rows start mid-document: nothing is masked


@pytest.mark.parametrize('attn_implementation', ['eager', 'sdpa'])
def test_packed_causal_batch_matches_per_document_forward(attn_implementation):
    model = _llama(attn_implementation)
    rows = [_docs([6, 4, 5]), _docs([7, 3])]  # the second row is shorter and gets a padding segment
    examples = [{'input_ids': [t for d in r for t in d], 'seq_lengths': [len(d) for d in r]} for r in rows]
    batch = PackedCausalLMCollator(pad_token_id=PAD)(examples)
    with torch.no_grad():
        # use_cache=False: with a KV cache present transformers skips the packed-sequence detection.
        # The varlen kwargs are passed along as the Trainer would; sdpa / eager ignore them.
        packed = model(
            input_ids=batch['input_ids'],
            position_ids=batch['position_ids'],
            cu_seq_lens_q=batch['cu_seq_lens_q'],
            cu_seq_lens_k=batch['cu_seq_lens_k'],
            max_length_q=batch['max_length_q'],
            max_length_k=batch['max_length_k'],
            use_cache=False,
        ).logits
        for r, row_docs in enumerate(rows):
            start = 0
            for doc in row_docs:
                single = model(input_ids=torch.tensor([doc]), use_cache=False).logits[0]
                torch.testing.assert_close(packed[r, start : start + len(doc)], single, atol=1e-4, rtol=1e-4)
                start += len(doc)
        # Without restarting position_ids the documents would attend to each other: the outputs must differ.
        naive = model(input_ids=batch['input_ids'], use_cache=False).logits
    assert not torch.allclose(packed[0, 6:10], naive[0, 6:10], atol=1e-4)


# ---------------------------------------------------------------- masked LM


def test_select_encoder_packing_mode():
    assert select_encoder_packing_mode('modernbert') == 'unpad'
    assert select_encoder_packing_mode('bert') == 'mask'
    assert select_encoder_packing_mode(None) == 'mask'
    assert select_encoder_packing_mode('bert', 'unpad') == 'unpad'
    with pytest.raises(ValueError):
        select_encoder_packing_mode('modernbert', 'mask')  # ModernBERT only takes 2D masks
    with pytest.raises(ValueError):
        select_encoder_packing_mode('bert', 'flash')


def test_mlm_unpad_mode_splits_rows_into_documents():
    tokenizer = _tokenizer()
    docs = _docs([6, 4, 5], with_special=True)
    packed = pack_tokenized_dataset(_dataset(docs, with_special_mask=True), seq_length=10, strategy='bfd')
    collator = PackedMaskedLMCollator(tokenizer=tokenizer, mlm_probability=1.0, mode='unpad')
    torch.manual_seed(0)
    batch = collator([packed[i] for i in range(len(packed))])
    assert batch['input_ids'].shape[0] == len(docs)
    assert sorted(batch['attention_mask'].sum(dim=1).tolist()) == sorted(len(d) for d in docs)
    # special tokens are never masked, everything else is (mlm_probability=1.0)
    for ids, labels, mask in zip(batch['input_ids'], batch['labels'], batch['attention_mask'], strict=True):
        n = int(mask.sum())
        assert labels[0] == -100 and labels[n - 1] == -100 and ids[0] == CLS and ids[n - 1] == SEP
        assert (labels[1 : n - 1] != -100).all()
        assert (labels[n:] == -100).all()


def test_mlm_mask_mode_builds_block_diagonal_mask_and_positions():
    tokenizer = _tokenizer()
    docs = _docs([4, 3], with_special=True)
    packed = pack_tokenized_dataset(_dataset(docs, with_special_mask=True), seq_length=8, strategy='bfd')
    assert len(packed) == 1 and packed[0]['seq_lengths'] == [4, 3]
    collator = PackedMaskedLMCollator(tokenizer=tokenizer, mlm_probability=0.0, mode='mask', pad_to_multiple_of=8)
    batch = collator([packed[0]])
    assert batch['input_ids'].shape == (1, 8)
    assert batch['position_ids'].tolist() == [[0, 1, 2, 3, 0, 1, 2, 0]]
    mask = batch['attention_mask'][0]
    assert mask.shape == (8, 8)
    assert mask[:4, :4].all() and mask[4:7, 4:7].all()
    assert not mask[:4, 4:].any() and not mask[4:7, :4].any() and not mask[7].any() and not mask[:, 7].any()


@pytest.mark.parametrize('attn_implementation', ['eager', 'sdpa'])
def test_packed_mlm_batch_matches_per_document_forward(attn_implementation):
    tokenizer = _tokenizer()
    model = _bert(attn_implementation)
    docs = _docs([6, 4, 5, 7, 3], with_special=True)  # packs into two rows of different lengths
    packed = pack_tokenized_dataset(_dataset(docs, with_special_mask=True), seq_length=15, strategy='bfd')
    assert len(packed) == 2
    batch = PackedMaskedLMCollator(tokenizer=tokenizer, mlm_probability=0.0, mode='mask')(
        [packed[i] for i in range(len(packed))]
    )
    with torch.no_grad():
        logits = model(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], position_ids=batch['position_ids']
        ).logits
        for r in range(len(packed)):
            start = 0
            for n in packed[r]['seq_lengths']:
                doc = batch['input_ids'][r, start : start + n].unsqueeze(0)
                single = model(input_ids=doc).logits[0]
                torch.testing.assert_close(logits[r, start : start + n], single, atol=1e-4, rtol=1e-4)
                start += n


# ---------------------------------------------------------------- data arguments


def test_data_arguments_validate_packing_options():
    mlm = MlmDataTrainingArguments(dataset_name='d', line_by_line=True, packing=True)
    assert mlm.packing_strategy == 'bfd' and mlm.packing_encoder_mode == 'auto'
    with pytest.raises(ValueError):
        MlmDataTrainingArguments(dataset_name='d', line_by_line=False, packing=True)
    with pytest.raises(ValueError):
        MlmDataTrainingArguments(dataset_name='d', line_by_line=True, packing=True, pad_to_max_length=True)
    with pytest.raises(ValueError):
        MlmDataTrainingArguments(dataset_name='d', line_by_line=True, packing=True, packing_strategy='random')
    with pytest.raises(ValueError):
        MlmDataTrainingArguments(dataset_name='d', line_by_line=True, packing=True, packing_encoder_mode='flash')
    clm = ClmDataTrainingArguments(dataset_name='d', packing=True, packing_seq_length=8192)
    assert clm.packing_mask_document_starts is True
    with pytest.raises(ValueError):
        ClmDataTrainingArguments(dataset_name='d', packing=True, packing_seq_length=0)
