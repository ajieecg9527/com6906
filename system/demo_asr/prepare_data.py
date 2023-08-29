""" Prepare data for the ASR model. """
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append("../../")
from data.utils import run_data_split
from model.hearing_loss.signal.msbg import run_msbg_simulation_for_signal
from model.hearing_loss.signal.process import run_signal_generation_train, run_signal_generation_test

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def prepare_data(cfg: DictConfig) -> None:
    """ Prepare data """

    # Load records
    train_set = Path(cfg.train_path.metadata_dir) / f"{cfg.dataset.train_set}.json"
    with train_set.open("r", encoding="utf-8") as fp:
        train_records = json.load(fp)

    test_set = Path(cfg.test_path.metadata_dir) / f"{cfg.dataset.test_set}.json"
    with test_set.open("r", encoding="utf-8") as fp:
        test_records = json.load(fp)

    # Run the MSBG simulation
    logger.info(f"Run MSGB simulation for {len(train_records)} hearing aid signals (train set).")
    for record in tqdm(train_records):
        signal_name = record["signal"]
        run_msbg_simulation_for_signal(signal_name, cfg["MSBGEar"], cfg.train_path, cfg.ref_sr)

    logger.info(f"Run MSGB simulation for {len(train_records)} hearing aid signals (test set).")
    for record in tqdm(test_records):
        signal_name = record["signal"]
        run_msbg_simulation_for_signal(signal_name, cfg["MSBGEar"], cfg.test_path, cfg.ref_sr)

    # Split the train set
    logger.info("Split the train set.")
    run_data_split(cfg.dataset.train_set, cfg.train_path, cfg.dev_percent)

    # Generate signals for ASR (train set)
    logger.info("Generate signals for ASR for cpc2_train_data")
    run_signal_generation_train(cfg.dataset.train_set, cfg.train_path, cfg.target_sr)

    # Generate signals for ASR (test set)
    logger.info("Generate signals for ASR for cpc2_test_data")
    run_signal_generation_test(cfg.dataset.test_set, cfg.test_path, cfg.target_sr)


if __name__ == "__main__":
    prepare_data()
