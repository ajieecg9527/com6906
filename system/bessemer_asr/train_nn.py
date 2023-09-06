import sys
import json
import logging
from pathlib import Path

import hydra
import torch
import pandas as pd
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from omegaconf import DictConfig

sys.path.append("../../")
from data.utils import generate_records_for_measure
from model.intelligibility.nn.nn import FeedforwardNeuralNetwork

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    hparams_file, run_opts, overrides = sb.parse_arguments([cfg.asr_config])
    with open(hparams_file, encoding="utf-8") as fp:
        hparams = load_hyperpyyaml(fp, overrides)

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
        train_scores = Path(hparams["haspi_folder"]) / f"{cfg.dataset.train_set}.{measure}.jsonl"
        test_scores = Path(hparams["haspi_folder"]) / f"{cfg.dataset.test_set}.{measure}.jsonl"
        # Add scores to the records
        train_records = generate_records_for_measure(measure, train_scores, train_records)
        test_records = generate_records_for_measure(measure, test_scores, test_records)

    # List -> Pandas
    labels = ["signal", "correctness"] + measures
    train_df, test_df = pd.DataFrame(train_records), pd.DataFrame(test_records)
    train_df, test_df = train_df[labels], test_df[labels]

    # Load encoder similarities
    with (Path(cfg.train_path.exp_dir) / f"train_enc_similarity.json").open("r", encoding="utf-8") as fp:
        train_encoder_similarity = json.load(fp)
    with (Path(cfg.train_path.exp_dir) / f"dev_enc_similarity.json").open("r", encoding="utf-8") as fp:
        dev_encoder_similarity = json.load(fp)
    with (Path(cfg.test_path.exp_dir) / f"test_enc_similarity.json").open("r", encoding="utf-8") as fp:
        test_encoder_similarity = json.load(fp)
    train_encoder_similarity.update(dev_encoder_similarity)  # merge

    # Load decoder similarities
    with (Path(cfg.train_path.exp_dir) / f"train_dec_similarity.json").open("r", encoding="utf-8") as fp:
        train_decoder_similarity = json.load(fp)
    with (Path(cfg.train_path.exp_dir) / f"dev_dec_similarity.json").open("r", encoding="utf-8") as fp:
        dev_decoder_similarity = json.load(fp)
    with (Path(cfg.test_path.exp_dir) / f"test_dec_similarity.json").open("r", encoding="utf-8") as fp:
        test_decoder_similarity = json.load(fp)
    train_decoder_similarity.update(dev_decoder_similarity)  # merge

    train_signal_list, test_signal_list = list(train_encoder_similarity.keys()), list(test_encoder_similarity.keys())
    # train_df = train_df[train_df["signal"].isin(train_signal_list)]

    # Add 2 more columns for similarities
    train_df["encoder_similarity"], train_df["decoder_similarity"] = 0, 0
    test_df["encoder_similarity"], test_df["decoder_similarity"] = 0, 0
    for signal in train_signal_list:
        train_df.loc[train_df["signal"] == signal, "encoder_similarity"] = train_encoder_similarity[signal]
        train_df.loc[train_df["signal"] == signal, "decoder_similarity"] = train_decoder_similarity[signal]
    for signal in test_signal_list:
        test_df.loc[test_df["signal"] == signal, "encoder_similarity"] = test_encoder_similarity[signal]
        test_df.loc[test_df["signal"] == signal, "decoder_similarity"] = test_decoder_similarity[signal]

    # Random split
    # Train : Dev : Test = 0.8 : 0.1 : 0.1
    dev_df = train_df.sample(frac=0.2, random_state=9527)
    train_df = pd.concat([train_df, dev_df]).drop_duplicates(keep=False)
    test_df = dev_df.sample(frac=0.5, random_state=9527)
    dev_df = pd.concat([dev_df, test_df]).drop_duplicates(keep=False)

    # Redefine labels
    similarities = ["encoder_similarity", "decoder_similarity"]
    labels = labels[:2] + similarities
    # labels = labels + similarities

    # Pandas -> Torch.Tensor
    train_set = torch.from_numpy(train_df[labels[1:]].astype("float32").values)
    dev_set = torch.from_numpy(dev_df[labels[1:]].astype("float32").values)
    test_set = torch.from_numpy(test_df[labels[1:]].astype("float32").values)

    # Build a Feedforward Neural Network
    nn = FeedforwardNeuralNetwork(n_inputs=len(similarities), n_neurons=32)
    # nn = FeedforwardNeuralNetwork(n_inputs=len(measures)+len(similarities), n_neurons=32)

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
    evaluate()
