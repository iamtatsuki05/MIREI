import logging
import os
import sys
from pathlib import Path
from typing import Any

import datasets
import fire
import torch
import transformers
from datasets import (
    ClassLabel,
    DatasetDict,
    IterableDatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    interleave_datasets,
    load_dataset,
)
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import SentenceEvaluator, TripletEvaluator
from transformers import HfArgumentParser, is_torch_xla_available, set_seed
from transformers.trainer_utils import get_last_checkpoint

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.train.st.data_class.data_training_arguments import DataTrainingArguments
from nlp.constract_llm.train.st.data_class.model_arguments import ModelArguments

logger = logging.getLogger(__name__)


def _setup_logging(training_args: SentenceTransformerTrainingArguments) -> None:
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


def _add_subset_label(
    ds: datasets.Dataset | datasets.IterableDataset,
    label_id: int,
    num_proc: int | None = None,
) -> datasets.Dataset | datasets.IterableDataset:
    return ds.map(
        lambda batch: {'label': [label_id] * len(batch[list(batch.keys())[0]])},
        batched=True,
        num_proc=num_proc,
        desc='Adding subset label',
    )


def _trim_split(
    ds_split: datasets.Dataset | datasets.IterableDataset,
    max_samples: int | None,
    streaming: bool,
    seed: int = 42,
):
    if max_samples is None:
        return ds_split

    max_samples = min(len(ds_split), max_samples)

    if streaming:
        buffer_size = max(max_samples * 2, 10_000)
        return ds_split.shuffle(buffer_size=buffer_size, seed=seed).take(max_samples)
    else:
        return ds_split.shuffle(seed=seed).select(range(max_samples))


def load_raw_datasets(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    training_args: SentenceTransformerTrainingArguments,
) -> DatasetDict | IterableDatasetDict:
    if data_args.dataset_name is None:
        data_files: dict[str, str] = {}
        if data_args.train_file:
            data_files['train'] = data_args.train_file
            extension = data_args.train_file.split('.')[-1]
        if data_args.validation_file:
            data_files['validation'] = data_args.validation_file
            extension = data_args.validation_file.split('.')[-1]
        if extension == 'txt':
            extension = 'text'
        return load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    if data_args.use_all_subset:
        subset_names = get_dataset_config_names(
            data_args.dataset_name,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
        )
    elif data_args.use_subsets:
        subset_names = list(data_args.use_subsets)
    else:
        subset_names = [data_args.dataset_config_name]

    label_feature = ClassLabel(names=subset_names)
    train_parts, valid_parts = [], []

    for cfg in subset_names:
        ds = load_dataset(
            data_args.dataset_name,
            cfg,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        label_id = label_feature.str2int(cfg)

        ds = {k: _add_subset_label(v, label_id, data_args.preprocessing_num_workers) for k, v in ds.items()}

        if 'validation' not in ds and training_args.do_eval:
            val_split = f'train[:{data_args.validation_split_percentage}%]'
            train_split = f'train[{data_args.validation_split_percentage}%:]'

            ds['validation'] = _add_subset_label(
                load_dataset(
                    data_args.dataset_name,
                    cfg,
                    split=val_split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                ),
                label_id,
                data_args.preprocessing_num_workers,
            )
            ds['train'] = _add_subset_label(
                load_dataset(
                    data_args.dataset_name,
                    cfg,
                    split=train_split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                ),
                label_id,
                data_args.preprocessing_num_workers,
            )

        for split_name in ('train', 'validation'):
            if split_name in ds and data_args.max_subset_samples is not None:
                ds[split_name] = _trim_split(
                    ds[split_name],
                    data_args.max_subset_samples,
                    data_args.streaming,
                    seed=training_args.seed,
                )

        train_parts.append(ds['train'])
        if training_args.do_eval and 'validation' in ds:
            valid_parts.append(ds['validation'])

    if data_args.streaming:
        data_dict = {'train': interleave_datasets(train_parts)}
        if valid_parts:
            data_dict['validation'] = interleave_datasets(valid_parts)
        raw_datasets = IterableDatasetDict(data_dict)
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].cast_column('label', label_feature)
    else:
        data_dict = {'train': concatenate_datasets(train_parts)}
        if valid_parts:
            data_dict['validation'] = concatenate_datasets(valid_parts)
        raw_datasets = DatasetDict(data_dict).cast_column('label', label_feature)

    return raw_datasets


def get_evaluator(
    eval_dataset: datasets.Dataset,
    evaluator_type: str,
    anchor_column_name: str | None = None,
    positive_column_name: str | None = None,
    negative_column_name: str | None = None,
) -> SentenceEvaluator | None:
    anchor_col = anchor_column_name or 'anchor'
    positive_col = positive_column_name or 'positive'
    negative_col = negative_column_name or 'negative'
    column_names = eval_dataset.column_names

    match evaluator_type:
        case 'triplet':
            required = {anchor_col, positive_col, negative_col}
            if not required.issubset(column_names):
                return None
            return TripletEvaluator(
                anchors=eval_dataset[anchor_col],
                positives=eval_dataset[positive_col],
                negatives=eval_dataset[negative_col],
                name='triplet-evaluator',
            )
        case _:
            raise ValueError(f'Unsupported evaluator type: {evaluator_type}')


def main(config_file_path: str | Path | None = None, **kwargs: Any) -> None:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            SentenceTransformerTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_dict(load_cli_config(config_file_path, **kwargs))

    # Setup logging
    _setup_logging(training_args)

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, '
        + f'distributed training: {training_args.parallel_mode.value == "distributed"}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info('Loading datasets...')
    raw_datasets = load_raw_datasets(data_args, model_args, training_args)
    logger.info(f'Raw datasets: {raw_datasets}')

    logger.info('Loading model...')
    common_kwargs = dict(
        cache_folder=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ('auto', None) else getattr(torch, model_args.torch_dtype)
    )
    model = SentenceTransformer(
        model_args.model_name_or_path,
        model_kwargs={
            'torch_dtype': torch_dtype,
            'attn_implementation': model_args.attn_implementation,
            'low_cpu_mem_usage': model_args.low_cpu_mem_usage,
        },
        tokenizer_kwargs={'use_fast': model_args.use_fast_tokenizer},
        **common_kwargs,
    )
    if data_args.max_seq_length is not None:
        model.max_seq_length = data_args.max_seq_length
    logger.info(f'SentenceTransformer model:\n{model}')

    logger.info('Preprocessing the datasets...')

    def prepare_features(examples):
        col_map = {
            'anchor': data_args.anchor_column_name,
            'positive': data_args.positive_column_name,
            'negative': data_args.negative_column_name,
            'label': data_args.label_column_name,
        }
        features = {k: examples[v] for k, v in col_map.items() if v}
        return features

    column_names = raw_datasets['train'].column_names
    required = [
        col
        for col in (
            data_args.anchor_column_name,
            data_args.positive_column_name,
            data_args.negative_column_name,
        )
        if col
    ]
    if training_args.batch_sampler == 'group_by_label':
        required.append('label')
    remove_columns = [col for col in column_names if col not in required]

    if training_args.do_train:
        train_dataset = (
            raw_datasets['train']
            .map(
                prepare_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=remove_columns,
                desc='Running preprocessing on train dataset',
            )
            .select_columns(required)
        )

    if training_args.do_eval:
        eval_dataset = (
            raw_datasets['validation']
            .map(
                prepare_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=remove_columns,
                desc='Running preprocessing on validation dataset',
            )
            .select_columns(required)
        )

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        # NOTE: SentenceTransformers is unsport for this https://github.com/UKPLab/sentence-transformers/issues/2888
        # metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        # def compute_metrics(eval_preds):
        #     preds, labels = eval_preds
        #     # preds have the same shape as the labels, after the argmax(-1) has been calculated
        #     # by preprocess_logits_for_metrics
        #     labels = labels.reshape(-1)
        #     preds = preds.reshape(-1)
        #     mask = labels != -100
        #     labels = labels[mask]
        #     preds = preds[mask]
        #     return metric.compute(predictions=preds, references=labels)

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f'train dataset: {train_dataset}')

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f'eval dataset: {eval_dataset}')

    loss = losses.CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=model_args.loss_cache_mini_batch_size or training_args.per_device_train_batch_size,
        scale=model_args.loss_scale,
    )

    evaluator = get_evaluator(
        eval_dataset=eval_dataset,
        evaluator_type=data_args.evaluator_type,
        anchor_column_name=data_args.anchor_column_name,
        positive_column_name=data_args.positive_column_name,
        negative_column_name=data_args.negative_column_name,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        loss=loss,
        evaluator=evaluator if training_args.do_eval and evaluator is not None else None,
        # compute_metrics=(compute_metrics if training_args.do_eval and not is_torch_xla_available() else None),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval and not is_torch_xla_available() else None
        ),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics['train_samples'] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

    logger.info('Saving model...')
    model.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        final_score = evaluator(model)
        logger.info(f'Final evaluation score: {final_score}')

    kwargs = {'finetuned_from': model_args.model_name_or_path, 'tasks': 'sentence-similarity'}
    if data_args.dataset_name is not None:
        kwargs['dataset_tags'] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs['dataset_args'] = data_args.dataset_config_name
            kwargs['dataset'] = f'{data_args.dataset_name} {data_args.dataset_config_name}'
        else:
            kwargs['dataset'] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == '__main__':
    fire.Fire(main)
