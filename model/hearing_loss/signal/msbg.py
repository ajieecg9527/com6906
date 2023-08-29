""" Run the MSBG hearing loss simulation model. """

import csv
import json
import logging
import math
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import DictConfig
from tqdm import tqdm
from numpy import ndarray

from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import MSBG_FS, pad
from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal, write_signal
from clarity.utils.signal_processing import resample

sys.path.append("../../")
from data.utils import parse_cec2_signal_name


def run_msbg_simulation_for_signal(signal_name, cfg_ear, path, ref_sr):
    """Run the MSBG simulation for a given signal."""

    # Initialize hearing loss ear
    ear = Ear(**cfg_ear)

    # Parse hearing aid signal name
    scene, listener_id, _ = parse_cec2_signal_name(signal_name)

    # Retrieve the listener
    listener_dict = Listener.load_listener_dict(Path(path.metadata_dir) / "listeners.json")
    listener = listener_dict[listener_id]

    # Read the hearing aid signal
    HA_signal = read_signal(Path(path.signal_dir) / f"{signal_name}.wav")

    # Resample (the msbg model can only deal with sample rate=44100)
    resampled_HA_signal = resample(HA_signal, sample_rate=ref_sr, new_sample_rate=MSBG_FS)

    # Process the hearing loss signal
    HL_signal = listen(ear, resampled_HA_signal, listener)

    # Write the hearing loss signal
    exp_signal_dir = Path(path.exp_dir) / "HL_outputs"
    exp_signal_dir.mkdir(parents=True, exist_ok=True)
    write_signal(
        exp_signal_dir / f"{signal_name}.wav",
        HL_signal,
        MSBG_FS,
        floating_point=True
    )


def listen(ear, signal, listener):
    """Generate MSBG processed signal."""

    ear.set_audiogram(listener.audiogram_left)
    out_l = ear.process(signal[:, 0])

    ear.set_audiogram(listener.audiogram_right)
    out_r = ear.process(signal[:, 1])

    if len(out_l[0]) != len(out_r[0]):
        diff = len(out_l[0]) - len(out_r[0])
        if diff > 0:
            out_r[0] = np.flipud(pad(np.flipud(out_r[0]), len(out_l[0])))
        else:
            out_l[0] = np.flipud(pad(np.flipud(out_l[0]), len(out_r[0])))

    return np.concatenate([out_l, out_r]).T
