""" Compute the HAAQI scores. """

import sys
from pathlib import Path
from scipy.io import wavfile

from clarity.evaluator.haaqi import haaqi_v1
from clarity.utils.audiogram import Listener

sys.path.append("../../")
from data.utils import parse_cec2_signal_name


def compute_haaqi_for_signal(signal_name, path, ref_suf="ref"):
    """Compute the HAAQI score for a given signal.

    Args:
        signal (str): name of the signal to process
        path (dict): paths to the signals and metadata, as defined in the config

    Returns:
        float: HAAQI score
    """

    scene, listener_id, _ = parse_cec2_signal_name(signal_name)

    # Retrieve the listeners
    listener_dict = Listener.load_listener_dict(Path(path["metadata_dir"]) / "listeners.json")
    listener = listener_dict[listener_id]

    # Retrieve signals and convert to float32 between -1 and 1
    sr_proc, proc = wavfile.read(Path(path["signal_dir"]) / f"{signal_name}.wav")
    sr_ref, ref = wavfile.read(Path(path["scene_dir"]) / f"{scene}_target_{ref_suf}.wav")
    assert sr_ref == sr_proc

    proc = proc / 32768.0
    ref = ref / 32768.0

    # Compute haaqi score using library code
    haaqi_left, _, _, _ = haaqi_v1(
        reference=ref[:, 0],
        reference_freq=sr_ref,
        processed=proc[:, 0],
        processed_freq=sr_proc,
        audiogram=listener.audiogram_left,
        level1=100,  # assume for HL listeners
    )

    haaqi_right, _, _, _ = haaqi_v1(
        reference=ref[:, 1],
        reference_freq=sr_ref,
        processed=proc[:, 1],
        processed_freq=sr_proc,
        audiogram=listener.audiogram_right,
        level1=100,  # assume for HL listeners
    )

    if haaqi_left > haaqi_right:
        haaqi, ear = haaqi_left, int(0)
    else:
        haaqi, ear = haaqi_right, int(1)

    return haaqi, ear