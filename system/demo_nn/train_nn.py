"""Train a feedforward neural network model."""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

sys.path.append("../../")
from data.utils import generate_records_for_measure
from model.intelligibility.nn.nn import FeedforwardNeuralNetwork

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def train_nn(cfg: DictConfig) -> None:
    """Train a feedforward neural network with 2 layers."""

    # Load records
    train_set = Path(cfg.train_path.metadata_dir) / f"{cfg.dataset.train_set}.json"
    with train_set.open("r", encoding="utf-8") as fp:
        train_records = json.load(fp)

    test_set = Path(cfg.test_path.metadata_dir) / f"{cfg.dataset.test_set}.json"
    with test_set.open("r", encoding="utf-8") as fp:
        test_records = json.load(fp)

    # Load scores and add them to the records
    measures = ["haspi", "hasqi", "haaqi"]
    for measure in measures:
        # Load scores
        train_scores = Path(cfg.train_path.exp_dir) / f"{cfg.dataset.train_set}.{measure}.jsonl"
        test_scores = Path(cfg.test_path.exp_dir) / f"{cfg.dataset.test_set}.{measure}.jsonl"
        # Add scores to the records
        train_records = generate_records_for_measure(measure, train_scores, train_records)
        test_records = generate_records_for_measure(measure, test_scores, test_records)

    # List -> Pandas
    labels = ["signal", "correctness"] + measures
    train_df, test_df = pd.DataFrame(train_records), pd.DataFrame(test_records)
    train_df, test_df = train_df[labels], test_df[labels]

    # Random split
    # Train : Dev : Test = 0.8 : 0.1 : 0.1
    dev_df = train_df.sample(frac=0.2, random_state=9527)
    train_df = pd.concat([train_df, dev_df]).drop_duplicates(keep=False)
    test_df = dev_df.sample(frac=0.5, random_state=9527)
    dev_df = pd.concat([dev_df, test_df]).drop_duplicates(keep=False)

    # Pandas -> Torch.Tensor
    train_set = torch.from_numpy(train_df[labels[1:]].astype("float32").values)
    dev_set = torch.from_numpy(dev_df[labels[1:]].astype("float32").values)
    test_set = torch.from_numpy(test_df[labels[1:]].astype("float32").values)

    # Build a Feedforward Neural Network
    nn = FeedforwardNeuralNetwork(n_inputs=len(measures), n_neurons=32)

    # Train
    nn.fit(train_set=train_set, dev_set=dev_set, epochs=10, lr=0.000001, batch_size=1)

    # Test
    predictions = nn.predict(test_set=test_set)
    predictions_df = test_df[["signal"]].rename(columns={"signal": "signal_ID"})
    predictions_df["intelligibility_score"] = predictions.detach().numpy()

    # Write
    results = Path(cfg.test_path.exp_dir) / f"{cfg.dataset.test_set}.predict.csv"
    predictions_df.to_csv(results, index=False)


if __name__ == "__main__":
    train_nn()
