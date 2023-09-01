""" Process the signals. """

import sys
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.utils.signal_processing import resample

sys.path.append("../../")
from data.utils import parse_cec2_signal_name, merge_csv_files


def run_signal_generation_train(dataset, path, target_sr):
    """ Genarate the MSBG signals for the train set. """

    dataset_json = Path(path.metadata_dir) / f"{dataset}.json"
    signal_train_json = Path(path.exp_dir) / f"data_split/{dataset}.train.json"
    signal_dev_json = Path(path.exp_dir) / f"data_split/{dataset}.dev.json"

    # Load json for signals
    with signal_train_json.open("r", encoding="utf-8") as fp:
        signal_train_list = json.load(fp)
    with signal_dev_json.open("r", encoding="utf-8") as fp:
        signal_dev_list = json.load(fp)
    with dataset_json.open("r", encoding="utf-8") as fp:
        records = json.load(fp)
    record_dict = {record["signal"]: record for record in records}

    # Generate the hearing aid signals (reference signals) in train set for ASR
    run_signal_resampling_and_cutting(
        path=path, dataset=dataset, signal_list=signal_train_list,
        record_dict=record_dict, split="train", _type="ref", target_sr=target_sr
    )

    # Generate the msbg hearing loss signals (processed signals) in train set for ASR
    run_signal_resampling_and_cutting(
        path=path, dataset=dataset, signal_list=signal_train_list,
        record_dict=record_dict, split="train", _type="msbg", target_sr=target_sr
    )

    # Generate the hearing aid signals (reference signals) in dev set for ASR
    run_signal_resampling_and_cutting(
        path=path, dataset=dataset, signal_list=signal_dev_list,
        record_dict=record_dict, split="dev", _type="ref", target_sr=target_sr
    )

    # Generate the msbg hearing loss signals (processed signals) in dev set for ASR
    run_signal_resampling_and_cutting(
        path=path, dataset=dataset, signal_list=signal_dev_list,
        record_dict=record_dict, split="dev", _type="msbg", target_sr=target_sr
    )

    # Merge the csv files
    target_dir = Path(path.exp_dir) / "cpc2_asr_data"
    train_csv_files = [target_dir / f"{dataset}.train.msbg.csv", target_dir / f"{dataset}.train.ref.csv"]
    dev_csv_files = [target_dir / f"{dataset}.dev.msbg.csv", target_dir / f"{dataset}.dev.ref.csv"]
    merge_csv_files(train_csv_files, target_dir / f"{dataset}.train.csv")
    merge_csv_files(dev_csv_files, target_dir / f"{dataset}.dev.csv")


def run_signal_generation_test(dataset, path, target_sr):
    """ Genarate the MSBG signals for the test set. """

    dataset_json = Path(path.metadata_dir) / f"{dataset}.json"

    # Load json for signals
    with dataset_json.open("r", encoding="utf-8") as fp:
        records = json.load(fp)
    record_dict = {record["signal"]: record for record in records}
    signal_test_list = [record["signal"] for record in records]

    # Generate the hearing aid signals (reference signals) in test set for ASR
    run_signal_resampling_and_cutting(
        path=path, dataset=dataset, signal_list=signal_test_list,
        record_dict=record_dict, split="test", _type="ref", target_sr=target_sr
    )

    # Generate the msbg hearing loss signals (processed signals) in test set for ASR
    run_signal_resampling_and_cutting(
        path=path, dataset=dataset, signal_list=signal_test_list,
        record_dict=record_dict, split="test", _type="msbg", target_sr=target_sr
    )

    # Merge the csv files
    target_dir = Path(path.exp_dir) / "cpc2_asr_data"
    test_csv_files = [target_dir / f"{dataset}.test.msbg.csv", target_dir / f"{dataset}.test.ref.csv"]
    merge_csv_files(test_csv_files, target_dir / f"{dataset}.test.csv")


def run_signal_resampling_and_cutting(path, dataset, signal_list, record_dict, split, _type, target_sr):
    """Resample and cut signals for ASR"""

    target_dir = Path(path.exp_dir) / "cpc2_asr_data"
    target_signal_dir = target_dir / f"{split}_{_type}"
    target_signal_dir.mkdir(parents=True, exist_ok=True)

    csv_lines = []
    target_csv = target_dir / f"{dataset}.{split}.{_type}.csv"

    signal_dir, wrds_label = None, None
    if _type == "ref":
        signal_dir = Path(path.signal_dir)
        wrds_label = "prompt"
    if _type == "msbg":
        signal_dir = Path(path.exp_dir) / f"HL_outputs"
        wrds_label = "response"

    for signal_name in tqdm(signal_list):
        signal = signal_dir / f"{signal_name}.wav"
        target_signal = target_signal_dir / f"{signal_name}.wav"

        scene, listener_id, _ = parse_cec2_signal_name(signal_name)
        wrds = record_dict[signal_name][wrds_label].upper()

        # Resample the signal
        utt, orig_sr = sf.read(signal)
        utt_16k = resample(utt, sample_rate=orig_sr, new_sample_rate=target_sr)

        # Get rid of the first 1 seconds, as there is no speech
        duration = len(utt_16k[1 * target_sr:, 0]) / target_sr
        sf.write(target_signal, utt_16k[1 * target_sr:, :], target_sr)
        csv_lines.append([signal_name, str(duration), str(target_signal), listener_id, wrds])

    df_lines = pd.DataFrame(csv_lines, columns=["signal_ID", "duration", "signal", "listener_id", "wrd"])
    IDs = [i for i in range(len(df_lines))]
    df_lines.insert(loc=0, column="ID", value=IDs)
    df_lines.to_csv(target_csv, index=False, sep=",")
