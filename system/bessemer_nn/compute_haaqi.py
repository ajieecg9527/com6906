""" This script computes the HAAQI scores. """
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.utils.file_io import write_jsonl

sys.path.append("../../")
from data.utils import set_seed_with_string
from model.hearing_loss.index.haaqi import compute_haaqi_for_signal

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def compute_haaqi(cfg: DictConfig) -> None:
    """Run the HAAQI score computation."""

    # Load records
    train_set = Path(cfg.train_path.metadata_dir) / f"{cfg.dataset.train_set}.json"
    with train_set.open("r", encoding="utf-8") as fp:
        train_records = json.load(fp)

    test_set = Path(cfg.test_path.metadata_dir) / f"{cfg.dataset.test_set}.json"
    with test_set.open("r", encoding="utf-8") as fp:
        test_records = json.load(fp)

    # Compute the haaqi scores and record the scores from the better ear
    logger.info(f"Computing scores for {len(train_records)} signals (train set).")
    Path(f"{cfg.train_path.exp_dir}").mkdir(parents=True, exist_ok=True)
    results_file = Path(f"{cfg.train_path.exp_dir}") / f"{cfg.dataset.train_set}.haaqi.jsonl"
    for record in tqdm(train_records):
        signal_name = record["signal"]

        # Set random seed for each signal
        set_seed_with_string(signal_name)

        # Get haaqi score and better ear
        haaqi, ear = compute_haaqi_for_signal(signal_name, cfg.train_path)

        # Write result
        result = {"signal": signal_name, "haaqi": haaqi, "ear": ear}
        write_jsonl(str(results_file), [result])

    # Compute the haaqi scores and record the scores from the better ear
    logger.info(f"Computing scores for {len(test_records)} signals (test set).")
    Path(f"{cfg.test_path.exp_dir}").mkdir(parents=True, exist_ok=True)
    results_file = Path(f"{cfg.test_path.exp_dir}") / f"{cfg.dataset.test_set}.haaqi.jsonl"
    for record in tqdm(test_records):
        signal_name = record["signal"]

        # Set random seed for each signal
        set_seed_with_string(signal_name)

        # Get haaqi score and better ear
        haaqi, ear = compute_haaqi_for_signal(signal_name, cfg.test_path, "anechoic")

        # Write result
        result = {"signal": signal_name, "haaqi": haaqi, "ear": ear}
        write_jsonl(str(results_file), [result])


if __name__ == "__main__":
    compute_haaqi()
