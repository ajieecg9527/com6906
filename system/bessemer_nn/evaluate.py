"""Evaluate the predictions against the ground truth correctness values"""
import sys
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import kendalltau, pearsonr

sys.path.append("../../")
from data.evaluator import compute_scores

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate the predictions against the ground truth correctness values"""

    # Load the intelligibility dataset records
    dataset_filename = Path(cfg.test_path.metadata_dir) / f"{cfg.dataset.test_set}.json"
    with open(dataset_filename, encoding="utf-8") as fp:
        records = json.load(fp)
    record_index = {record["signal"]: record for record in records}

    # Load the predictions
    predictions = Path(cfg.test_path.exp_dir) / f"{cfg.dataset.test_set}.predict.csv"
    df = pd.read_csv(
        predictions, names=["signal", "predicted"], header=0
    )

    df["correctness"] = [record_index[signal]["correctness"] for signal in df.signal]

    # Compute and report the scores
    scores = compute_scores(df["predicted"], df["correctness"])

    results = Path(cfg.test_path.exp_dir) / f"{cfg.dataset.test_set}.evaluate.jsonl"
    with open(results, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(scores) + "\n")

    # Output the scores to the console
    print(scores)


if __name__ == "__main__":
    evaluate()
