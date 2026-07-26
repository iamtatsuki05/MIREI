#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning the library models for extractive question answering (SQuAD / JSQuAD)."""
# Adapted from transformers v4.56.1 examples/pytorch/question-answering/run_qa.py.

import logging
import os
import sys
from pathlib import Path
from typing import Any

import datasets
import evaluate
import fire
import numpy as np
import transformers
from datasets import load_dataset
from pydantic import Field
from pydantic.dataclasses import dataclass
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from mirei.common.utils.cli_utils import load_cli_config
from mirei.constract_llm.eval.jglue.data_class.model_arguments import ModelArguments
from mirei.constract_llm.eval.qa.trainer_qa import QuestionAnsweringTrainer
from mirei.constract_llm.eval.qa.utils_qa import postprocess_qa_predictions

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    task_name: str | None = Field(default=None, metadata={'help': 'Task label for bookkeeping (e.g. squad, JSQuAD).'})
    dataset_name: str | None = Field(default=None, metadata={'help': 'HF dataset repository.'})
    dataset_config_name: str | None = Field(default=None, metadata={'help': 'HF dataset config name.'})
    dataset_revision: str | None = Field(default=None, metadata={'help': 'HF dataset revision.'})
    max_seq_length: int = Field(default=384, metadata={'help': 'Maximum total input sequence length.'})
    doc_stride: int = Field(default=128, metadata={'help': 'Stride between chunks when splitting long documents.'})
    n_best_size: int = Field(default=20, metadata={'help': 'Number of n-best predictions to generate.'})
    max_answer_length: int = Field(default=30, metadata={'help': 'Maximum answer length in tokens.'})
    version_2_with_negative: bool = Field(default=False, metadata={'help': 'Whether unanswerable questions exist.'})
    null_score_diff_threshold: float = Field(default=0.0, metadata={'help': 'Threshold for null answer selection.'})
    pad_to_max_length: bool = Field(default=False, metadata={'help': 'Pad all samples to max_seq_length.'})
    max_train_samples: int | None = Field(default=None, metadata={'help': 'Truncate the train set for debugging.'})
    max_eval_samples: int | None = Field(default=None, metadata={'help': 'Truncate the eval set for debugging.'})
    preprocessing_num_workers: int | None = Field(default=None, metadata={'help': 'Workers for preprocessing.'})
    overwrite_cache: bool = Field(default=False, metadata={'help': 'Overwrite cached datasets.'})


def _setup_logging(training_args: TrainingArguments) -> int:
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return log_level


def _ensure_pad_token(tokenizer: Any, model: Any) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('Tokenizer has neither a pad token nor an EOS token.')
        logger.warning('Tokenizer does not have a pad token. Using EOS token as pad token.')
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


def _assert_right_padding(tokenizer: Any) -> None:
    padding_probe = tokenizer(
        ['padding probe', 'a longer padding probe'],
        padding='max_length',
        max_length=16,
        truncation=True,
        return_attention_mask=True,
    )
    if tokenizer.padding_side != 'right' or any(
        mask[0] != 1 or mask[-1] != 0 for mask in padding_probe['attention_mask']
    ):
        raise RuntimeError(
            f'Right-padding preflight failed: padding_side={tokenizer.padding_side} '
            f'attention_mask={padding_probe["attention_mask"]}'
        )
    logger.info('tokenizer_padding_side=right padding_preflight=passed')


class FiniteMetricQATrainer(QuestionAnsweringTrainer):
    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        non_finite_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (float, np.floating)) and not np.isfinite(value)
        }
        if non_finite_metrics:
            raise RuntimeError(f'Non-finite evaluation metrics: {non_finite_metrics}')
        return metrics


def main(config_file_path: str | Path | None = None, **kwargs: Any) -> None:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(load_cli_config(config_file_path, **kwargs))

    log_level = _setup_logging(training_args)
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        revision=data_args.dataset_revision,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.padding_side = 'right'
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool('.ckpt' in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        attn_implementation=model_args.attn_implementation,
    )

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError('This script requires a fast tokenizer (offset mapping support).')

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    _ensure_pad_token(tokenizer, model)
    _assert_right_padding(tokenizer)

    column_names = (
        raw_datasets['train'].column_names if training_args.do_train else raw_datasets['validation'].column_names
    )
    question_column_name = 'question' if 'question' in column_names else column_names[0]
    context_column_name = 'context' if 'context' in column_names else column_names[1]
    answer_column_name = 'answers' if 'answers' in column_names else column_names[2]

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f'max_seq_length ({data_args.max_seq_length}) > model max length ({tokenizer.model_max_length}); clipping.'
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def prepare_train_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name],
            examples[context_column_name],
            truncation='only_second',
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length' if data_args.pad_to_max_length else False,
        )
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        offset_mapping = tokenized_examples.pop('offset_mapping')

        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples['input_ids'][i]
            if tokenizer.cls_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.cls_token_id)
            elif tokenizer.bos_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.bos_token_id)
            else:
                cls_index = 0
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers['answer_start']) == 0:
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                # Keep the answer search inside the context span. Upstream relies on a
                # (0, 0)-offset special token to stop the walk, which tokenizers that
                # add no special tokens (e.g. sarashina) do not provide.
                context_start_index = token_start_index
                context_end_index = token_end_index
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                else:
                    while token_start_index <= context_end_index and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples['start_positions'].append(token_start_index - 1)
                    while token_end_index >= context_start_index and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples['end_positions'].append(token_end_index + 1)
        return tokenized_examples

    def prepare_validation_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name],
            examples[context_column_name],
            truncation='only_second',
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length' if data_args.pad_to_max_length else False,
        )
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        tokenized_examples['example_id'] = []
        for i in range(len(tokenized_examples['input_ids'])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            tokenized_examples['example_id'].append(examples['id'][sample_index])
            tokenized_examples['offset_mapping'][i] = [
                (o if sequence_ids[k] == 1 else None) for k, o in enumerate(tokenized_examples['offset_mapping'][i])
            ]
        return tokenized_examples

    if training_args.do_train:
        train_dataset = raw_datasets['train']
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        with training_args.main_process_first(desc='train dataset map pre-processing'):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc='Running tokenizer on train dataset',
            )

    eval_examples = raw_datasets['validation']
    if data_args.max_eval_samples is not None:
        eval_examples = eval_examples.select(range(min(len(eval_examples), data_args.max_eval_samples)))
    with training_args.main_process_first(desc='validation dataset map pre-processing'):
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc='Running tokenizer on validation dataset',
        )

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    def post_processing_function(examples, features, predictions, stage='eval'):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        formatted_predictions = [{'id': str(k), 'prediction_text': v} for k, v in predictions.items()]
        references = [{'id': str(ex['id']), 'answers': ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load(
        'squad_v2' if data_args.version_2_with_negative else 'squad', cache_dir=model_args.cache_dir
    )

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    trainer = FiniteMetricQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        metrics = trainer.evaluate()
        metrics['eval_samples'] = len(eval_dataset)
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)


if __name__ == '__main__':
    fire.Fire(main)
