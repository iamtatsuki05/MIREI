from __future__ import annotations

import torch
from peft import PeftModel
from torch import nn
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import is_torch_flex_attn_available, logging

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)


class SentenceEncoderMixin:
    @staticmethod
    def _skip_instruction(sentence_feature: BatchEncoding):
        assert sentence_feature['attention_mask'].shape == sentence_feature['embed_mask'].shape
        sentence_feature['attention_mask'] = sentence_feature['embed_mask']

    @staticmethod
    def _tokenize_with_instruction(
        sentences: str | list[str],
        tokenizer: AutoTokenizer,
        instruction: str = '',
        max_length: int | None = None,
    ) -> BatchEncoding:
        if isinstance(sentences, str):
            sentences = [sentences]

        plain = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False,
            max_length=max_length,
        )
        mixed = tokenizer(
            [instruction + s for s in sentences],
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=max_length,
        )

        masks = []
        for i, seq_len in enumerate(plain['attention_mask'].sum(dim=1)):
            m = torch.zeros_like(mixed['attention_mask'][i])
            if seq_len > 0:
                m[-seq_len:] = 1
            masks.append(m.unsqueeze(0))
        mixed['embed_mask'] = torch.cat(masks, dim=0)
        return mixed

    def _pool(self, feats: BatchEncoding, hidden: torch.Tensor) -> torch.Tensor:
        if self.skip_instruction:
            self._skip_instruction(feats)

        seq_lens = feats['attention_mask'].sum(dim=-1)

        if self.pooling_mode == 'mean':
            return torch.stack(
                [
                    (hidden[i, -l:, :].mean(dim=0) if l > 0 else hidden.new_zeros(hidden.size(-1)))
                    for i, l in enumerate(seq_lens)
                ],
                dim=0,
            )
        elif self.pooling_mode == 'weighted_mean':
            bs, L, _ = hidden.shape
            w = hidden.new_zeros(bs, L)
            for i, l in enumerate(seq_lens):
                if l > 0:
                    ar = torch.arange(l, device=w.device) + 1
                    w[i, -l:] = ar / ar.sum()
            return (hidden * w.unsqueeze(-1)).sum(dim=1)
        elif self.pooling_mode in ('eos_token', 'last_token'):
            return hidden[:, -1]
        elif self.pooling_mode == 'bos_token':
            bos_id = self.tokenizer.bos_token_id
            mask = (feats['input_ids'] == bos_id).unsqueeze(-1).expand_as(hidden)
            return hidden[mask].view(hidden.size(0), -1)
        else:
            raise ValueError(f'{self.pooling_mode} not supported.')

    @torch.no_grad()
    def encode(
        self,
        sentences: str | list[str],
        tokenizer: AutoTokenizer,
        *,
        instruction: str = '',
        max_length: int | None = None,
    ) -> torch.Tensor:
        self.tokenizer = tokenizer
        feats = self._tokenize_with_instruction(
            sentences,
            tokenizer,
            instruction=instruction,
            max_length=max_length,
        ).to(self.device)
        hidden = self(**feats).last_hidden_state
        return self._pool(feats, hidden)


class LlamaBiModel(LlamaModel, SentenceEncoderMixin):
    _no_split_modules = ['LlamaDecoderLayer']

    def __init__(
        self,
        config: LlamaConfig,
        *,
        pooling_mode: str = 'mean',
        skip_instruction: bool = True,
        max_length: int | None = None,
    ) -> None:
        super().__init__(config)
        self.set_causal(False)

        self.pooling_mode = pooling_mode
        self.skip_instruction = skip_instruction
        self.tokenizer: AutoTokenizer | None = None
        self.max_length = max_length

    def set_causal(self, causal: bool = False):
        for layer in self.layers:
            layer.self_attn.is_causal = causal

    def _update_causal_mask(
        self,
        attention_mask: 'BlockMask' | torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == 'flash_attention_2':
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == 'flex_attention':
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == 'sdpa'
            and attention_mask is not None
            and attention_mask.device.type in ['cuda', 'xpu', 'npu']
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        # device: torch.device, this not used
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        device = cache_position.device
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            # causal_mask = torch.full(
            #     (sequence_length, target_length),
            #     fill_value=min_dtype,
            #     dtype=dtype,
            #     device=device,
            # )

            # if sequence_length != 1:
            #     causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = torch.zeros((sequence_length, target_length), dtype=dtype, device=device)

            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def encode_document(
        self,
        sentences: str | list[str],
        tokenizer: AutoTokenizer,
        instruction: str = '',
        max_length: int | None = None,
    ) -> torch.Tensor:
        return self.encode(sentences, tokenizer, instruction=instruction, max_length=max_length)

    def encode_query(
        self,
        sentences: str | list[str],
        tokenizer: AutoTokenizer,
        instruction: str = '',
        max_length: int | None = None,
    ) -> torch.Tensor:
        return self.encode(
            sentences,
            tokenizer,
            instruction=instruction,
            max_length=max_length,
        )


class LlamaBiForMNTP(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        pooling_mode: str = 'mean',
        skip_instruction: bool = True,
        max_length: int | None = None,
    ) -> None:
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaBiModel(
            config,
            pooling_mode=pooling_mode,
            skip_instruction=skip_instruction,
            max_length=max_length,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @torch.no_grad()
    def encode(
        self,
        sentences: str | list[str],
        tokenizer: AutoTokenizer,
        *,
        instruction: str = '',
        max_length: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model.encode(
            sentences,
            tokenizer,
            instruction=instruction,
            max_length=max_length,
            **kwargs,
        )

    def set_causal(self, causal: bool = False):
        self.model.set_causal(causal)

    @property
    def pooling_mode(self):
        return self.model.pooling_mode

    @pooling_mode.setter
    def pooling_mode(self, mode: str):
        self.model.pooling_mode = mode

    @property
    def skip_instruction(self):
        return self.model.skip_instruction

    @skip_instruction.setter
    def skip_instruction(self, flag: bool):
        self.model.skip_instruction = flag

    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    def save_peft_model(self, path: str):
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        else:
            raise ValueError('self.model is not a PeftModel')


class LlamaBiForSequenceClassification(LlamaPreTrainedModel):
    def __init__(
        self,
        config: LlamaConfig,
        pooling_mode: str = 'mean',
        skip_instruction: bool = True,
        max_length: int | None = None,
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaBiModel(
            config,
            pooling_mode=pooling_mode,
            skip_instruction=skip_instruction,
            max_length=max_length,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError('Cannot handle batch sizes > 1 if no padding token is defined.')
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be '
                'unexpected if using padding tokens in conjunction with `inputs_embeds.`'
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                pooled_logits=pooled_logits,
                config=self.config,
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
