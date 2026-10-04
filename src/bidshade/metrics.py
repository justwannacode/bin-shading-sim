"""Landscape quality metrics."""

from __future__ import annotations

import numpy as np

from .dataset import AuctionData
from .landscape.base import Landscape

EPS = 1e-6


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    y = y.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y.astype(np.float64)) ** 2))


def anlp(landscape: Landscape, data: AuctionData, bin_width: float, chunk_size: int) -> float:
    """Average negative log probability of the true market price bin.

    P(z in [lo, lo + w)) = W(lo + w) - W(lo). Standard metric in bid landscape papers;
    it scores the whole distribution, not just win/loss at one bid.
    """
    lower = np.floor(data.z / bin_width) * bin_width
    bids = np.column_stack([lower, lower + bin_width])
    w = landscape.predict(data.X, data.segment, bids, chunk_size=chunk_size)
    p = np.clip(w[:, 1] - w[:, 0], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict[str, np.ndarray]:
    """Uniform-width reliability bins on [0, 1]; empty bins are dropped."""
    bin_id = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(bin_id, minlength=n_bins)
    sum_p = np.bincount(bin_id, weights=p, minlength=n_bins)
    sum_y = np.bincount(bin_id, weights=y.astype(np.float64), minlength=n_bins)
    keep = counts > 0
    return {
        "mean_pred": sum_p[keep] / counts[keep],
        "frac_won": sum_y[keep] / counts[keep],
        "count": counts[keep],
    }


def expected_calibration_error(table: dict[str, np.ndarray]) -> float:
    counts = table["count"]
    gap = np.abs(table["mean_pred"] - table["frac_won"])
    return float(np.sum(counts * gap) / np.sum(counts))
