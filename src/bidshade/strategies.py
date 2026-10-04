"""Bidding strategies compared in the experiment."""

from __future__ import annotations

import numpy as np

from .dataset import AuctionData
from .simulator import first_price_auction


def truthful_bids(data: AuctionData) -> np.ndarray:
    """Bid the full value: wins everything, earns zero surplus in a first-price auction."""
    return data.v.copy()


def fixed_shading_bids(data: AuctionData, k: float) -> np.ndarray:
    return k * data.v


def oracle_bids(data: AuctionData, eps: float = 0.01) -> np.ndarray:
    """Knows z in advance: bids just above it when profitable. Upper bound on surplus."""
    target = data.z + eps
    return np.where(target < data.v, target, 0.0)


def fixed_k_surplus_curve(data: AuctionData, k_grid: np.ndarray) -> np.ndarray:
    """Total surplus (CPM units) of fixed shading for every k in the grid."""
    return np.array([first_price_auction(data.v, data.z, k * data.v).surplus.sum() for k in k_grid])


def tune_fixed_k(data: AuctionData, k_grid: np.ndarray) -> tuple[float, np.ndarray]:
    """Offline tuning of k on validation logs (in production this would be an A/B test)."""
    curve = fixed_k_surplus_curve(data, k_grid)
    return float(k_grid[int(np.argmax(curve))]), curve
