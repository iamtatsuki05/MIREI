"""Document-boundary-aware sequence packing shared by MLM (encoder) and CLM (decoder) pre-training.

Packing itself is delegated to ``trl.data_utils.pack_dataset`` (best-fit decreasing: documents are never split
across rows; each row carries ``seq_lengths``, the length of every document in the row). The collators below turn
those boundaries into what each model family understands, so that attention never crosses a document:

* decoder (Llama / Qwen / Mistral ...): ``position_ids`` restart at every document and **no** ``attention_mask`` is
  emitted. transformers (>= 4.53) detects the restarts and builds block-diagonal masks for flash_attention_2 (varlen),
  sdpa and eager; the model must run with ``use_cache=False`` (a KV cache disables that detection). Padding is
  appended as one more segment whose labels are ``-100``.
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
from typing import Any, Literal

import torch
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

PackingStrategy = Literal['bfd', 'wrapped']
EncoderPackingMode = Literal['auto', 'unpad', 'mask']
PACKING_STRATEGIES: tuple[str, ...] = ('bfd', 'wrapped')
ENCODER_PACKING_MODES: tuple[str, ...] = ('auto', 'unpad', 'mask')
# Encoders whose forward removes padding and builds cu_seqlens itself; feeding them one document per row is enough.
UNPADDING_ENCODER_MODEL_TYPES: frozenset[str] = frozenset({'modernbert'})
SEQ_LENGTHS_COLUMN = 'seq_lengths'


def _truncate_lists(example: dict[str, list[Any]], seq_length: int) -> dict[str, list[Any]]:
    return {k: (v[:seq_length] if isinstance(v, list) else v) for k, v in example.items()}


def pack_tokenized_dataset(
    dataset: Dataset | DatasetDict,
    seq_length: int,
    strategy: str = 'bfd',
    num_proc: int | None = None,
    load_from_cache_file: bool = True,
) -> Dataset | DatasetDict:
    """Pack an already tokenized dataset (``input_ids`` plus any parallel list columns) into rows of ``seq_length``.

    Documents longer than ``seq_length`` are truncated first because ``bfd`` never splits a document. Every list
    column (``attention_mask``, ``special_tokens_mask``, ...) is packed in parallel with ``input_ids``.
    """
    if strategy not in PACKING_STRATEGIES:
        raise ValueError(f'Unsupported packing_strategy={strategy!r}; expected one of {PACKING_STRATEGIES}')
    if seq_length <= 0:
        raise ValueError(f'packing seq_length must be positive, got {seq_length}')
    from trl.data_utils import pack_dataset

    if strategy == 'bfd':
        # bfd never splits a document, so anything longer than a row has to be truncated first
        dataset = dataset.map(
            _truncate_lists,
            fn_kwargs={'seq_length': seq_length},
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            desc=f'Truncating documents to {seq_length} tokens before packing',
        )
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


@dataclass
class PackedCausalLMCollator:
    """Collator for packed causal LM rows.

    Emits ``input_ids``, ``position_ids`` (restarting per document) and ``labels``; no ``attention_mask`` so that
    transformers derives the block-diagonal mask from ``position_ids``. Padding forms its own trailing segment with
    ``labels=-100``. With ``mask_document_starts`` the first token of every document is excluded from the loss
    (predicting it from the previous document is meaningless), mirroring ``DataCollatorWithFlattening``.
    """

    pad_token_id: int
    mask_document_starts: bool = True
    pad_to_multiple_of: int | None = None

    def __call__(self, examples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = [(torch.as_tensor(ex['input_ids'], dtype=torch.long), _segment_lengths(ex)) for ex in examples]
        max_len = max(ids.numel() for ids, _ in rows)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        input_ids, position_ids, labels = [], [], []
        for ids, lengths in rows:
            pad = max_len - ids.numel()
            lab = ids.clone()
            if self.mask_document_starts:
                starts = torch.cumsum(torch.tensor([0] + lengths[:-1]), dim=0)
                lab[starts] = -100
            seg_lengths = lengths + ([pad] if pad else [])
            input_ids.append(torch.cat([ids, ids.new_full((pad,), self.pad_token_id)]))
            position_ids.append(segment_position_ids(seg_lengths))
            labels.append(torch.cat([lab, lab.new_full((pad,), -100)]))
        return {
            'input_ids': torch.stack(input_ids),
            'position_ids': torch.stack(position_ids),
            'labels': torch.stack(labels),
        }


def select_encoder_packing_mode(model_type: str | None, mode: str = 'auto') -> str:
    """Resolve ``auto`` to ``unpad`` for encoders that unpad internally (ModernBERT) and ``mask`` otherwise."""
    if mode not in ENCODER_PACKING_MODES:
        raise ValueError(f'Unsupported packing_encoder_mode={mode!r}; expected one of {ENCODER_PACKING_MODES}')
    if mode != 'auto':
        return mode
    return 'unpad' if model_type in UNPADDING_ENCODER_MODEL_TYPES else 'mask'


@dataclass
class PackedMaskedLMCollator:
    """Collator for packed MLM rows; masking is delegated to ``DataCollatorForLanguageModeling``.

    * ``mode='unpad'``: every packed row is split back into its documents, which become separate (padded) batch
      entries. Models with internal unpadding (ModernBERT + flash_attention_2) re-concatenate them with
      ``cu_seqlens``; any other attention implementation simply sees a per-document batch.
    * ``mode='mask'``: the packed row stays as is; ``attention_mask`` becomes a 3D block-diagonal mask and
      ``position_ids`` restart per document, so BERT-style encoders never attend across documents.
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
    def _split_row(example: Mapping[str, Any]) -> list[dict[str, list[Any]]]:
        lengths = _segment_lengths(example)
        docs: list[dict[str, list[Any]]] = []
        start = 0
        for n in lengths:
            doc = {
                k: list(v[start : start + n])
                for k, v in example.items()
                if k != SEQ_LENGTHS_COLUMN and isinstance(v, (list, tuple)) and len(v) == len(example['input_ids'])
            }
            docs.append(doc)
            start += n
        return docs

    def __call__(self, examples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if self.mode == 'unpad':
            docs = [doc for ex in examples for doc in self._split_row(ex)]
            return self._mlm(docs)
        rows = [
            {k: list(v) for k, v in ex.items() if k != SEQ_LENGTHS_COLUMN and isinstance(v, (list, tuple))}
            for ex in examples
        ]
        batch = self._mlm(rows)
        seq_len = batch['input_ids'].shape[1]
        position_ids, masks = [], []
        for ex in examples:
            lengths = _segment_lengths(ex)
            pad = seq_len - sum(lengths)
            seg = segment_ids(lengths + ([pad] if pad else []))
            valid = torch.cat([torch.ones(sum(lengths), dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
            position_ids.append(segment_position_ids(lengths + ([pad] if pad else [])))
            masks.append(block_diagonal_attention_mask(seg, valid))
        batch['position_ids'] = torch.stack(position_ids)
        batch['attention_mask'] = torch.stack(masks)
        return batch
