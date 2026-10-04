"""Turn logged auctions into the feedback a first-price bidder would actually observe.

For every logged request we place a random test bid b' = alpha * v and replay the auction
against the logged competing price z:
  * b' > z  -> win, z is observed (exact event);
  * b' <= z -> loss, we only know z >= b' (right-censored at b').
Landscape models must be trained on CensoredSample only: it never exposes z for losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import AuctionData


@dataclass
class CensoredSample:
    X: np.ndarray
    segment: np.ndarray
    v: np.ndarray
    bid: np.ndarray
    won: np.ndarray
    price_obs: np.ndarray  # z for wins, NaN for losses
    feature_names: list[str]
    categorical_features: list[str]

    def __len__(self) -> int:
        return len(self.bid)

    @property
    def durations(self) -> np.ndarray:
        """Survival-analysis view: event time (z) for wins, censoring time (b') for losses."""
        return np.where(self.won, self.price_obs, self.bid)

    @property
    def win_rate(self) -> float:
        return float(self.won.mean()) if len(self) else float("nan")

    def take(self, idx: np.ndarray) -> CensoredSample:
        return CensoredSample(
            X=self.X[idx],
            segment=self.segment[idx],
            v=self.v[idx],
            bid=self.bid[idx],
            won=self.won[idx],
            price_obs=self.price_obs[idx],
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
        )

    def sample(self, n_max: int | None, seed: int = 0) -> CensoredSample:
        if n_max is None or len(self) <= n_max:
            return self
        idx = np.sort(np.random.default_rng(seed).choice(len(self), size=n_max, replace=False))
        return self.take(idx)


def make_censored_sample(
    data: AuctionData,
    alpha_low: float,
    alpha_high: float,
    replicates: int = 1,
    seed: int = 0,
) -> CensoredSample:
    if not 0.0 <= alpha_low < alpha_high:
        raise ValueError(f"Need 0 <= alpha_low < alpha_high, got {alpha_low}, {alpha_high}")
    if replicates < 1:
        raise ValueError("replicates must be >= 1")

    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(len(data)), replicates)
    alpha = rng.uniform(alpha_low, alpha_high, size=len(idx))
    bid = alpha * data.v[idx]
    z = data.z[idx]
    won = bid > z
    return CensoredSample(
        X=data.X[idx],
        segment=data.segment[idx],
        v=data.v[idx],
        bid=bid,
        won=won,
        price_obs=np.where(won, z, np.nan),
        feature_names=data.feature_names,
        categorical_features=data.categorical_features,
    )
