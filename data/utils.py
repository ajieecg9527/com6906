from __future__ import annotations

import json
import math
import random
import hashlib
from pathlib import Path

import numpy as np

from clarity.utils.file_io import read_jsonl


def set_seed_with_string(seed_string):
    """Set the random seed with a string."""
    md5_int = int(hashlib.md5(seed_string.encode("utf-8")).hexdigest(), 16) % (10**8)
    np.random.seed(md5_int)


def parse_cec2_signal_name(signal_name):
    """Parse the CEC2 signal name."""
    # e.g. S0001_L0001_E001_hr -> S0001, L0001, E001_hr
    scene, listener, *system = signal_name.split("_")
    if scene == "" or listener == "" or system == []:
        raise ValueError(f"Invalid CEC2 signal name: {signal_name}")
    return scene, listener, "_".join(system)


def generate_records_for_measure(measure, score_file, records):
    """Add scores to the records from one objective intelligibility measurement."""
    score = read_jsonl(score_file)
    score_index = {record["signal"]: record[f"{measure}"] for record in score}
    for record in records:
        record[f"{measure}"] = score_index[record["signal"]]
    return records


def run_data_split(dataset, path, dev_percent):
    """Split train set into train set and dev set according to dev_percent"""

    data_split_dir = Path(path.exp_dir) / "data_split"
    data_split_dir.mkdir(parents=True, exist_ok=True)
    signal_train_json = data_split_dir / f"{dataset}.train.json"
    signal_dev_json = data_split_dir / f"{dataset}.dev.json"

    # Read
    dataset_json = Path(path.metadata_dir) / f"{dataset}.json"
    with dataset_json.open("r", encoding="utf-8") as fp:
         records = json.load(fp)
    scene_list = [record["signal"] for record in records]

    # Split
    signal_dev_list = random.sample(scene_list, math.floor(len(scene_list) * dev_percent))
    signal_train_list = list(set(scene_list) - set(signal_dev_list))

    # Write
    with signal_train_json.open("w", encoding="utf-8") as fp:
        json.dump(signal_train_list, fp)
    with signal_dev_json.open("w", encoding="utf-8") as fp:
        json.dump(signal_dev_list, fp)
