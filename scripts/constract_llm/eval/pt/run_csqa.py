#!/usr/bin/env python
"""Fine-tuning for five-way multiple choice (CommonsenseQA / JCommonsenseQA).

Each (question, choice) pair is scored with a single-logit sequence-classification
head (architecture-native pooling: CLS for encoders, last non-pad token for
decoders), and the five scores are trained with a softmax cross-entropy over
choices. This matches the math of `AutoModelForMultipleChoice`, which is not
available for decoder architectures.
"""

import logging
import os
import sys
from dataclasses import dataclass as py_dataclass
from pathlib import Path
from typing import Any

import datasets
import evaluate
import fire
import numpy as np
import torch
import transformers
from datasets import load_dataset
from pydantic import Field
from pydantic.dataclasses import dataclass
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from mirei.common.utils.cli_utils import load_cli_config
from mirei.constract_llm.eval.jglue.data_class.model_arguments import ModelArguments

logger = logging.getLogger(__name__)

NUM_CHOICES = 5


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    task_name: str | None = Field(default=None, metadata={'help': 'Task label for bookkeeping (e.g. csqa, JCSQA).'})
    dataset_name: str | None = Field(default=None, metadata={'help': 'HF dataset repository.'})
    dataset_config_name: str | None = Field(default=None, metadata={'help': 'HF dataset config name.'})
    dataset_revision: str | None = Field(default=None, metadata={'help': 'HF dataset revision.'})
    max_seq_length: int = Field(default=128, metadata={'help': 'Maximum total input sequence length per choice.'})
    max_train_samples: int | None = Field(default=None, metadata={'help': 'Truncate the train set for debugging.'})
    max_eval_samples: int | None = Field(default=None, metadata={'help': 'Truncate the eval set for debugging.'})
    preprocessing_num_workers: int | None = Field(default=None, metadata={'help': 'Workers for preprocessing.'})
    overwrite_cache: bool = Field(default=False, metadata={'help': 'Overwrite cached datasets.'})


def _setup_logging(training_args: TrainingArguments) -> None:
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


def _extract_choices(example: dict[str, Any]) -> tuple[str, list[str], int]:
    question = example['question']
    if 'choices' in example:  # tau/commonsense_qa
        texts = example['choices']['text']
        labels = example['choices']['label']
        answer = example['answerKey']
        label = labels.index(answer)
    else:  # shunk031/JGLUE JCommonsenseQA
        texts = [example[f'choice{i}'] for i in range(NUM_CHOICES)]
        label = int(example['label'])
    if len(texts) != NUM_CHOICES:
        raise ValueError(f'Expected {NUM_CHOICES} choices, got {len(texts)}')
    return question, texts, label


@py_dataclass
class DataCollatorForMultipleChoice:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [f.pop('label') for f in features]
        batch_size = len(features)
        flattened = [
            {key: feature[key][i] for key in ('input_ids', 'attention_mask') if key in feature}
            for feature in features
            for i in range(NUM_CHOICES)
        ]
        batch = self.tokenizer.pad(flattened, padding=True, return_tensors='pt')
        batch = {key: value.view(batch_size, NUM_CHOICES, -1) for key, value in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch


class MultipleChoiceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        batch_size, num_choices, seq_len = inputs['input_ids'].shape
        flat_inputs = {key: value.reshape(batch_size * num_choices, seq_len) for key, value in inputs.items()}
        outputs = model(**flat_inputs)
        logits = outputs.logits.reshape(batch_size, num_choices)
        loss = nn.functional.cross_entropy(logits, labels)
        return (loss, {'logits': logits}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs['labels']
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, dict(inputs), return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), outputs['logits'].detach(), labels)

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

    _setup_logging(training_args)
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
        num_labels=1,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.padding_side = 'right'
    model = AutoModelForSequenceClassification.from_pretrained(
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

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    _ensure_pad_token(tokenizer, model)
    _assert_right_padding(tokenizer)

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        keys = list(examples.keys())
        num_examples = len(examples[keys[0]])
        rows = [{key: examples[key][i] for key in keys} for i in range(num_examples)]
        questions: list[str] = []
        choices: list[str] = []
        labels: list[int] = []
        for row in rows:
            question, texts, label = _extract_choices(row)
            questions.extend([question] * NUM_CHOICES)
            choices.extend(texts)
            labels.append(label)
        tokenized = tokenizer(questions, choices, truncation=True, max_length=max_seq_length)
        grouped = {
            key: [values[i : i + NUM_CHOICES] for i in range(0, len(values), NUM_CHOICES)]
            for key, values in tokenized.items()
        }
        grouped['label'] = labels
        return grouped

    processed: dict[str, Any] = {}
    for split in ('train', 'validation'):
        dataset = raw_datasets[split]
        max_samples = data_args.max_train_samples if split == 'train' else data_args.max_eval_samples
        if max_samples is not None:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
        with training_args.main_process_first(desc=f'{split} dataset map pre-processing'):
            processed[split] = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f'Running tokenizer on {split} dataset',
            )

    accuracy = evaluate.load('accuracy', cache_dir=model_args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        predictions = np.argmax(p.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=p.label_ids)

    trainer = MultipleChoiceTrainer(
        model=model,
        args=training_args,
        train_dataset=processed['train'] if training_args.do_train else None,
        eval_dataset=processed['validation'] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
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
        metrics['train_samples'] = len(processed['train'])
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        metrics = trainer.evaluate()
        metrics['eval_samples'] = len(processed['validation'])
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)


if __name__ == '__main__':
    fire.Fire(main)
