"""Bid optimization: b* = argmax_b (v - b) * W(b | x) over a grid of bids."""

from __future__ import annotations

import numpy as np

from .dataset import AuctionData
from .landscape.base import Landscape


def make_q_grid(q_min: float, grid_size: int) -> np.ndarray:
    """Candidate bids as shares of the value v."""
    if not 0.0 <= q_min < 1.0:
        raise ValueError(f"q_min must be in [0, 1), got {q_min}")
    return np.linspace(q_min, 1.0, grid_size)


def argmax_surplus(
    v: np.ndarray, bids: np.ndarray, win_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the surplus-maximizing bid per row. Rows with no positive expected surplus get 0."""
    expected = (v[:, None] - bids) * win_prob
    j = np.argmax(expected, axis=1)
    rows = np.arange(len(v))
    best_bid, best_surplus = bids[rows, j], expected[rows, j]
    no_bid = best_surplus <= 0
    return np.where(no_bid, 0.0, best_bid), np.where(no_bid, 0.0, best_surplus)


def optimal_bids(
    landscape: Landscape,
    data: AuctionData,
    q_grid: np.ndarray,
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (bids, expected surplus under the model) for every request in `data`."""
    n = len(data)
    bids_out = np.empty(n, dtype=np.float64)
    surplus_out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        sl = slice(start, start + chunk_size)
        v = data.v[sl]
        grid = v[:, None] * q_grid[None, :]
        w = landscape.win_prob(data.X[sl], data.segment[sl], grid)
        bids_out[sl], surplus_out[sl] = argmax_surplus(v, grid, w)
    return bids_out, surplus_out
