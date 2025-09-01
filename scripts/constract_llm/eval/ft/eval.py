import logging
from pathlib import Path
from typing import Any

import fire
import torch
from sentence_transformers import SentenceTransformer

from nlp.common.utils.cli_utils import load_cli_config
from nlp.common.utils.file.json import save_as_indented_json
from nlp.constract_llm.eval.sentence_model.config import CLIConfig
from nlp.constract_llm.eval.sentence_model.dataset import prepare_dataset
from nlp.constract_llm.eval.sentence_model.metric import compute_alignment, compute_uniformity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_and_encode(cfg: CLIConfig):
    """
    Common setup: seed, device, model, dataset, output dir.
    Returns: model, positive_pairs, random_pairs, out_dir
    """
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positive_pairs, random_pairs = prepare_dataset(
        cfg.num_examples,
        cfg.miracl_name,
        cfg.miracl_lang,
        cfg.wiki_name,
        cfg.wiki_lang,
    )
    model_id = cfg.model_name_or_path
    logger.info(f'Model: {model_id}')
    model = SentenceTransformer(model_id).to(device)
    out_dir = Path(cfg.output_dir) / 'alignment_and_uniformity' / model_id.replace('/', '_')
    out_dir.mkdir(parents=True, exist_ok=True)
    return model, positive_pairs, random_pairs, out_dir


def alignment(config_file_path: str | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    model, positive_pairs, _, out_dir = setup_and_encode(cfg)

    z1 = model.encode([pair[0] for pair in positive_pairs], convert_to_tensor=True)
    z2 = model.encode([pair[1] for pair in positive_pairs], convert_to_tensor=True)
    alignment_score = compute_alignment(z1, z2)
    logger.info(f'Alignment:  {alignment_score:.4f}')
    save_as_indented_json({'alignment': alignment_score}, out_dir / 'result.json')


def uniformity(config_file_path: str | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    model, _, random_pairs, out_dir = setup_and_encode(cfg)

    z1 = model.encode([pair[0] for pair in random_pairs], convert_to_tensor=True)
    z2 = model.encode([pair[1] for pair in random_pairs], convert_to_tensor=True)
    uniformity_score = compute_uniformity(torch.cat([z1, z2], dim=0))
    logger.info(f'Uniformity: {uniformity_score:.4f}')

    save_as_indented_json({'uniformity': uniformity_score}, out_dir / 'result.json')


def main(config_file_path: str | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    model, positive_pairs, random_pairs, out_dir = setup_and_encode(cfg)

    z1 = model.encode([pair[0] for pair in positive_pairs], convert_to_tensor=True)
    z2 = model.encode([pair[1] for pair in positive_pairs], convert_to_tensor=True)
    alignment_score = compute_alignment(z1, z2)

    z1_rand = model.encode([pair[0] for pair in random_pairs], convert_to_tensor=True)
    z2_rand = model.encode([pair[1] for pair in random_pairs], convert_to_tensor=True)
    uniformity_score = compute_uniformity(torch.cat([z1_rand, z2_rand], dim=0))

    logger.info(f'Alignment:  {alignment_score:.4f}')
    logger.info(f'Uniformity: {uniformity_score:.4f}')

    save_as_indented_json({'alignment': alignment_score, 'uniformity': uniformity_score}, out_dir / 'result.json')


if __name__ == '__main__':
    fire.Fire({'main': main, 'alignment': alignment, 'uniformity': uniformity})
