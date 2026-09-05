"""Document-boundary-aware sequence packing shared by MLM (encoder) and CLM (decoder) pre-training.

Packing itself is delegated to ``trl.data_utils.pack_dataset`` (best-fit decreasing: documents are never split
across rows, documents longer than a row are truncated; each row carries ``seq_lengths``, the length of every
document in the row). The collators below turn those boundaries into what each model family understands, so that
attention never crosses a document:

* decoder (Llama / Qwen / Mistral ...): ``position_ids`` restart at every document, **no** ``attention_mask`` is
  emitted and the FlashAttention varlen kwargs (``cu_seq_lens_q/k``, ``max_length_q/k``) are attached. With
  flash_attention_2 the varlen kwargs drive the kernel directly (batches of any size); with sdpa / eager transformers
  (>= 4.53, torch >= 2.6) derives a block-diagonal mask from the ``position_ids`` restarts, which only happens when
  no KV cache is in play (``use_cache=False``). Padding is appended as one more segment whose labels are ``-100``.
* encoder with unpadding (ModernBERT): rows are split back into documents and handed to the standard MLM collator;
  ModernBERT removes the padding and concatenates the documents with ``cu_seqlens`` (true packing with
  flash_attention_2, padded per-document batches otherwise).
* other encoders (BERT / RoBERTa ...): the packed row is kept and a 3D block-diagonal ``attention_mask`` plus
  per-document ``position_ids`` are emitted (``get_extended_attention_mask`` accepts 3D masks).

``packing_strategy='wrapped'`` reproduces the classic concatenate-and-chunk behaviour (documents are split and no
``seq_lengths`` column exists); the collators then treat every row as a single segment.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from datasets.features import Sequence
from packaging import version
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

PACKING_STRATEGIES: tuple[str, ...] = ('bfd', 'wrapped')
ENCODER_PACKING_MODES: tuple[str, ...] = ('auto', 'unpad', 'mask')
# Encoders whose forward removes padding and builds cu_seqlens itself; feeding them one document per row is enough.
# Their forward only accepts a 2D attention_mask, so the 'mask' mode cannot be used with them.
UNPADDING_ENCODER_MODEL_TYPES: frozenset[str] = frozenset({'modernbert'})
SEQ_LENGTHS_COLUMN = 'seq_lengths'
MIN_TRL_VERSION = '0.18.0'  # pack_dataset(strategy='bfd') and the seq_lengths column
MIN_TRANSFORMERS_VERSION = '4.53.0'  # packed-sequence detection from position_ids for sdpa / eager
MIN_TORCH_VERSION = '2.6.0'  # masking_utils applies the packed mask only from this version on


def check_packing_requirements(attn_implementation: str | None = None) -> None:
    """Fail fast instead of silently training without document boundaries on too old libraries."""
    import transformers
    import trl

    if version.parse(trl.__version__) < version.parse(MIN_TRL_VERSION):
        raise RuntimeError(f'packing needs trl>={MIN_TRL_VERSION} (found {trl.__version__})')
    if version.parse(transformers.__version__) < version.parse(MIN_TRANSFORMERS_VERSION):
        raise RuntimeError(
            f'packing needs transformers>={MIN_TRANSFORMERS_VERSION} (found {transformers.__version__})'
        )
    if attn_implementation != 'flash_attention_2' and version.parse(torch.__version__) < version.parse(
        MIN_TORCH_VERSION
    ):
        raise RuntimeError(
            f'packing with attn_implementation={attn_implementation!r} needs torch>={MIN_TORCH_VERSION} '
            f'(found {torch.__version__}); transformers ignores the packed mask on older torch'
        )


def _keep_sequence_columns(dataset: Dataset) -> Dataset:
    """Drop scalar columns (ids, titles, ...): ``pack_dataset`` can only pack list columns."""
    drop = [name for name, feature in dataset.features.items() if not isinstance(feature, Sequence | list)]
    if drop:
        logger.info(f'dropping non-sequence columns before packing: {drop}')
        dataset = dataset.remove_columns(drop)
    return dataset


def pack_tokenized_dataset(
    dataset: Dataset | DatasetDict,
    seq_length: int,
    strategy: str = 'bfd',
    num_proc: int | None = None,
    load_from_cache_file: bool = True,
) -> Dataset | DatasetDict:
    """Pack an already tokenized dataset (``input_ids`` plus parallel list columns) into rows of ``seq_length``.

    ``bfd`` truncates documents longer than ``seq_length`` and keeps every document in one row; every list column
    (``attention_mask``, ``special_tokens_mask``, ...) is packed in parallel with ``input_ids``. Scalar columns are
    dropped.
    """
    if strategy not in PACKING_STRATEGIES:
        raise ValueError(f'Unsupported packing_strategy={strategy!r}; expected one of {PACKING_STRATEGIES}')
    if seq_length <= 0:
        raise ValueError(f'packing seq_length must be positive, got {seq_length}')
    check_packing_requirements()
    from trl.data_utils import pack_dataset

    if isinstance(dataset, DatasetDict):
        dataset = DatasetDict({split: _keep_sequence_columns(ds) for split, ds in dataset.items()})
    else:
        dataset = _keep_sequence_columns(dataset)
    packed = pack_dataset(
        dataset,
        seq_length,
        strategy=strategy,
        map_kwargs={'num_proc': num_proc, 'load_from_cache_file': load_from_cache_file},
    )
    if isinstance(packed, DatasetDict):
        for split, ds in packed.items():
            logger.info(f'packed {split}: {len(ds)} rows of <= {seq_length} tokens (strategy={strategy})')
    else:
        logger.info(f'packed dataset: {len(packed)} rows of <= {seq_length} tokens (strategy={strategy})')
    return packed


def _segment_lengths(example: Mapping[str, Any]) -> list[int]:
    """Document lengths of one packed row; a row without ``seq_lengths`` (wrapped strategy) is one segment."""
    n_tokens = len(example['input_ids'])
    lengths = list(example.get(SEQ_LENGTHS_COLUMN) or [n_tokens])
    if sum(lengths) != n_tokens:
        raise ValueError(f'seq_lengths {lengths} do not sum to the row length {n_tokens}')
    return [int(x) for x in lengths]


def segment_position_ids(lengths: list[int]) -> torch.Tensor:
    """``position_ids`` that restart at 0 for every segment, e.g. [3, 2] -> [0, 1, 2, 0, 1]."""
    if not lengths:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat([torch.arange(n, dtype=torch.long) for n in lengths])


def segment_ids(lengths: list[int]) -> torch.Tensor:
    """Segment index of every token, e.g. [3, 2] -> [0, 0, 0, 1, 1]."""
    if not lengths:
        return torch.zeros(0, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(len(lengths), dtype=torch.long), torch.tensor(lengths))


def block_diagonal_attention_mask(seg: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(L, L) mask that lets a token attend only to valid tokens of its own segment."""
    same = seg[:, None] == seg[None, :]
    return (same & valid[:, None] & valid[None, :]).to(torch.long)


def flash_attention_varlen_kwargs(rows: list[list[int]]) -> dict[str, Any]:
    """``cu_seq_lens_q/k`` and ``max_length_q/k`` over a batch flattened row-major, one entry per segment.

    ``_flash_attention_forward`` reshapes ``(batch, seq)`` to ``(batch * seq,)`` before calling the varlen kernel,
    so the offsets simply continue from one row into the next.
    """
    lengths = [n for row in rows for n in row]
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0)
    max_length = max(lengths) if lengths else 0
    return {'cu_seq_lens_q': cu, 'cu_seq_lens_k': cu, 'max_length_q': max_length, 'max_length_k': max_length}


@dataclass
class PackedCausalLMCollator:
    """Collator for packed causal LM rows.

    Emits ``input_ids``, ``position_ids`` (restarting per document), ``labels`` and, unless disabled, the
    FlashAttention varlen kwargs; no ``attention_mask`` so that sdpa / eager derive the block-diagonal mask from
    ``position_ids``. Padding forms its own trailing segment with ``labels=-100``. With ``mask_document_starts`` the
    first token of every document is excluded from the loss (predicting it from the previous document is
    meaningless), mirroring ``DataCollatorWithFlattening``; rows without boundaries (wrapped strategy) are untouched.
    """

    pad_token_id: int
    mask_document_starts: bool = True
    return_flash_attn_kwargs: bool = True
    pad_to_multiple_of: int | None = None

    def __call__(self, examples: list[Mapping[str, Any]]) -> dict[str, Any]:
        rows = [
            (torch.as_tensor(ex['input_ids'], dtype=torch.long), _segment_lengths(ex), SEQ_LENGTHS_COLUMN in ex)
            for ex in examples
        ]
        max_len = max(ids.numel() for ids, _, _ in rows)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        input_ids, position_ids, labels, seg_rows = [], [], [], []
        for ids, lengths, has_boundaries in rows:
            pad = max_len - ids.numel()
            lab = ids.clone()
            if self.mask_document_starts and has_boundaries:
                starts = torch.cumsum(torch.tensor([0] + lengths[:-1]), dim=0)
                lab[starts] = -100
            seg_lengths = lengths + ([pad] if pad else [])
            input_ids.append(torch.cat([ids, ids.new_full((pad,), self.pad_token_id)]))
            position_ids.append(segment_position_ids(seg_lengths))
            labels.append(torch.cat([lab, lab.new_full((pad,), -100)]))
            seg_rows.append(seg_lengths)
        batch: dict[str, Any] = {
            'input_ids': torch.stack(input_ids),
            'position_ids': torch.stack(position_ids),
            'labels': torch.stack(labels),
        }
        if self.return_flash_attn_kwargs:
            batch.update(flash_attention_varlen_kwargs(seg_rows))
        return batch


def select_encoder_packing_mode(model_type: str | None, mode: str = 'auto') -> str:
    """Resolve ``auto`` to ``unpad`` for encoders that unpad internally (ModernBERT) and ``mask`` otherwise."""
    if mode not in ENCODER_PACKING_MODES:
        raise ValueError(f'Unsupported packing_encoder_mode={mode!r}; expected one of {ENCODER_PACKING_MODES}')
    unpadding = model_type in UNPADDING_ENCODER_MODEL_TYPES
    if mode == 'mask' and unpadding:
        raise ValueError(
            f"model_type={model_type!r} only accepts 2D attention masks; use packing_encoder_mode='unpad'"
        )
    if mode != 'auto':
        return mode
    return 'unpad' if unpadding else 'mask'


@dataclass
class PackedMaskedLMCollator:
    """Collator for packed MLM rows; masking is delegated to ``DataCollatorForLanguageModeling``.

    * ``mode='unpad'``: every packed row is split back into its documents, which become separate (padded) batch
      entries. Models with internal unpadding (ModernBERT + flash_attention_2) re-concatenate them with
      ``cu_seqlens``; any other attention implementation simply sees a per-document batch.
    * ``mode='mask'``: the packed row stays as is; ``attention_mask`` becomes a 3D block-diagonal mask and
      ``position_ids`` restart per document, so BERT-style encoders never attend across documents. Tokens marked 0
      in the row's own ``attention_mask`` (padding inside a document) are excluded as well.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    mode: str = 'unpad'
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ('unpad', 'mask'):
            raise ValueError(f"mode must be 'unpad' or 'mask', got {self.mode!r}")
        self._mlm = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    @staticmethod
    def _list_columns(example: Mapping[str, Any]) -> dict[str, list[Any]]:
        n = len(example['input_ids'])
        return {
            k: list(v)
            for k, v in example.items()
            if k != SEQ_LENGTHS_COLUMN and isinstance(v, list | tuple) and len(v) == n
        }

    @classmethod
    def _split_row(cls, example: Mapping[str, Any]) -> list[dict[str, list[Any]]]:
        columns = cls._list_columns(example)
        docs: list[dict[str, list[Any]]] = []
        start = 0
        for n in _segment_lengths(example):
            docs.append({k: v[start : start + n] for k, v in columns.items()})
            start += n
        return docs

    def __call__(self, examples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if self.mode == 'unpad':
            return self._mlm([doc for ex in examples for doc in self._split_row(ex)])
        rows = [self._list_columns(ex) for ex in examples]
        batch = self._mlm(rows)
        seq_len = batch['input_ids'].shape[1]
        position_ids, masks = [], []
        for ex, padded_mask in zip(examples, batch['attention_mask'], strict=True):
            lengths = _segment_lengths(ex)
            pad = seq_len - sum(lengths)
            seg = segment_ids(lengths + ([pad] if pad else []))
            position_ids.append(segment_position_ids(lengths + ([pad] if pad else [])))
            masks.append(block_diagonal_attention_mask(seg, padded_mask.bool()))
        batch['position_ids'] = torch.stack(position_ids)
        batch['attention_mask'] = torch.stack(masks)
        return batch
