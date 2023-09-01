""" Train an ASR model. """

import sys
import logging
from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

sys.path.append("../../")
from model.intelligibility.asr.asr import dataio_prepare, ASR

logger = logging.getLogger(__name__)


def train_asr():
    # Input the hyperparameter file
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fp:
        hparams = load_hyperpyyaml(fp, overrides)

    # If distributed_launch is True
    # then create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts=run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create the datasets and the tokenizer
    train_set = hparams["train_set"]
    train_csv, dev_csv = hparams["train_csv"], hparams["valid_csv"]
    tokenizer, bos_index, eos_index = hparams["tokenizer"], hparams["bos_index"], hparams["eos_index"]

    haspi_file = Path(hparams["haspi_folder"]) / f"{train_set}.haspi.jsonl"
    train_set = dataio_prepare(train_csv, haspi_file, tokenizer, bos_index, eos_index)
    dev_set = dataio_prepare(dev_csv, haspi_file, tokenizer, bos_index, eos_index)

    # Download the pretrained language model from HuggingFace,
    # or elsewhere depending on the path given in the yaml file
    run_on_main(hparams["pretrainer"].collect_files)
    if run_opts["device"] == "cuda:0":  # if running on a CPU-only machine
        run_opts["device"] = "cpu"
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Build the ASR model
    # asr_model = ASR(
    #     modules=hparams["modules"],
    #     opt_class=hparams["Adam"],
    #     hparams=hparams,
    #     run_opts=run_opts,
    #     checkpointer=hparams["checkpointer"]
    # )
    #
    # # Set the tokenizer
    # asr_model.set_tokenizer(tokenizer=tokenizer)
    #
    # # Stage: train and dev
    # asr_model.fit(
    #     epoch_counter=hparams["epoch_counter"],
    #     train_set=train_set,
    #     valid_set=dev_set,
    #     train_loader_kwargs=hparams["train_dataloader_opts"],
    #     valid_loader_kwargs=hparams["valid_dataloader_opts"]
    # )


if __name__ == "__main__":
    train_asr()
