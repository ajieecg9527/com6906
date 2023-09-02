import sys
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.append("../../")
from data.evaluator import compute_scores
from data.utils import load_pred_and_label
from model.intelligibility.nn.lr import Logistic

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def evaluate(cfg: DictConfig) -> None:

    # Load the encoder representation
    dev_prediction, dev_label = load_pred_and_label(Path(cfg.train_path.exp_dir) / f"dev_enc_similarity.json",
                                                    Path(cfg.train_path.metadata_dir) / f"{cfg.dataset}.json")

    test_prediction, test_label = load_pred_and_label(Path(cfg.test_path.exp_dir) / f"dev_enc_similarity.json",
                                                      Path(cfg.test_path.metadata_dir) / f"{cfg.dataset}.json")

    logger.info("Apply logistic fitting.")
    model = Logistic()
    model.fit(dev_prediction, dev_label)
    fit_prediction = model.predict(test_prediction)
    enc_scores = compute_scores(fit_prediction, dev_label)

    # Load the decoder representation
    dev_prediction, dev_label = load_pred_and_label(Path(cfg.train_path.exp_dir) / f"dev_dec_similarity.json",
                                                    Path(cfg.train_path.metadata_dir) / f"{cfg.dataset}.json")

    test_prediction, test_label = load_pred_and_label(Path(cfg.test_path.exp_dir) / f"dev_dec_similarity.json",
                                                      Path(cfg.test_path.metadata_dir) / f"{cfg.dataset}.json")

    logger.info("Apply logistic fitting.")
    model = Logistic()
    model.fit(dev_prediction, dev_label)
    fit_prediction = model.predict(test_prediction)
    dec_scores = compute_scores(fit_prediction, test_label)

    results_file = Path(cfg.path.exp_dir) / "results.json"
    with results_file.open("w", encoding="utf-8") as fp:
        json.dump({"enc_results": enc_scores, "dec_results": dec_scores}, fp)


if __name__ == "__main__":
    evaluate()
