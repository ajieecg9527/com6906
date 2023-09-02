"""Evaluate the predictions against the ground truth correctness values"""

import numpy as np
from scipy.stats import kendalltau, pearsonr


def rmse_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the root mean squared error between two arrays"""
    return np.sqrt(np.mean((x - y) ** 2))


def ncc_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the normalized cross correlation between two arrays"""
    return pearsonr(x, y)[0]


def kt_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Kendall's tau correlation between two arrays"""
    return kendalltau(x, y)[0]


def std_err(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the standard error between two arrays"""
    return np.std(x - y) / np.sqrt(len(x))


def compute_scores(predictions, labels) -> dict:
    """Compute the scores for the predictions"""
    return {
        "RMSE": rmse_score(predictions, labels),
        "Std": std_err(predictions, labels),
        "NCC": ncc_score(predictions, labels),
        "KT": kt_score(predictions, labels),
    }