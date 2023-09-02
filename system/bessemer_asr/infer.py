""" Infer the similarity. """

import sys
import json
import logging
from pathlib import Path

import hydra
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from omegaconf import DictConfig
from speechbrain.utils.distributed import run_on_main
from tqdm import tqdm

sys.path.append("../../")
from model.intelligibility.asr.asr import ASR, compute_similarity

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def infer(cfg: DictConfig) -> None:
    # Load the trained ASR model
    hparams_file, run_opts, overrides = sb.parse_arguments([cfg.asr_config])
    with open(hparams_file, encoding="utf-8") as fp:
        hparams = load_hyperpyyaml(fp, overrides)

    tokenizer, bos_index = hparams["tokenizer"], hparams["bos_index"]

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    if run_opts["device"] == "cuda:0":  # if running on a CPU-only machine
        run_opts["device"] = "cpu"
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Initialize the ASR model
    asr_model = ASR(modules=hparams["modules"], opt_class=hparams["Adam"],
                    hparams=hparams, run_opts=run_opts, checkpointer=hparams["checkpointer"])
    asr_model.init_evaluation()

    # Load csv files
    train_exp_dir, test_exp_dir = Path(cfg.train_path.exp_dir), Path(cfg.test_path.exp_dir)
    dev_msbg_csv = sb.dataio.dataio.load_data_csv(train_exp_dir / f"cpc2_asr_data/{cfg.dataset.train_set}.dev.msbg.csv")
    test_msbg_csv = sb.dataio.dataio.load_data_csv(test_exp_dir / f"cpc2_asr_data/{cfg.dataset.test_set}.test.msbg.csv")

    # Compute the similarity on the dev set
    dev_enc_similarity, dev_dec_similarity = {}, {}
    for i, record in tqdm(dev_msbg_csv.items()):
        sig_msbg, wrd = record["signal"], record["wrd"]
        similarity = compute_similarity(sig_msbg, wrd, asr_model, bos_index, tokenizer)
        dev_enc_similarity[i] = similarity[0].tolist()
        dev_dec_similarity[i] = similarity[1].tolist()

        with (train_exp_dir / "dev_enc_similarity.json").open("w", encoding="utf-8") as fp:
            json.dump(dev_enc_similarity, fp)
        with (test_exp_dir / "dev_dec_similarity.json").open("w", encoding="utf-8") as fp:
            json.dump(dev_dec_similarity, fp)

    # Compute the similarity on the test set
    test_enc_similarity, test_dec_similarity = {}, {}
    test_dec_similarity = {}
    for i, record in tqdm(test_msbg_csv.items()):
        sig_msbg, wrd = record["signal"], record["wrd"]
        similarity = compute_similarity(sig_msbg, wrd, asr_model, bos_index, tokenizer)
        test_enc_similarity[i] = similarity[0].tolist()
        test_dec_similarity[i] = similarity[1].tolist()

        with (train_exp_dir / "test_enc_similarity.json").open("w", encoding="utf-8") as fp:
            json.dump(test_enc_similarity, fp)
        with (test_exp_dir / "test_dec_similarity.json").open("w", encoding="utf-8") as fp:
            json.dump(test_dec_similarity, fp)


if __name__ == "__main__":
    infer()