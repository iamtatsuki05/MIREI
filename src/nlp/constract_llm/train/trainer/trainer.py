import logging
from pathlib import Path

import torch
from transformers import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNTPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ['labels']

    def _remove_unused_columns(self, dataset, description: str | None = None):
        return dataset

    # We need a custom save function as we have to save the inner model
    def _save(self, output_dir: str | None = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving model checkpoint to {output_dir}')

        # model organization is MODEL_TYPEBiForMNTP.model -> MODEL_TYPELBiModel, we have to save the inner model, handled by save_peft_model function of the outer model
        # self.model.save_peft_model(str(output_dir))
        try:
            self.model.save_peft_model(str(output_dir))
        except Exception as e:
            logger.warning(f'Error saving peft model: {e}')
            logger.info('Saving the inner model instead')
            self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, str(output_dir / 'training_args.bin'))
