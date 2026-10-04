"""Common interface for bid landscape models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..censoring import CensoredSample


class Landscape(ABC):
    """A model of W(b | x) = P(z < b | x): the probability to win with bid b."""

    name: str = "landscape"
    fit_seconds: float = float("nan")

    @abstractmethod
    def fit(self, train: CensoredSample, valid: CensoredSample | None = None) -> Landscape:
        """Fit on censored feedback only."""

    @abstractmethod
    def _win_prob_2d(self, X: np.ndarray, segment: np.ndarray, bids: np.ndarray) -> np.ndarray:
        """bids has shape (n, g); returns W of the same shape."""

    def win_prob(self, X: np.ndarray, segment: np.ndarray, bids: np.ndarray) -> np.ndarray:
        """W(b | x) for bids of shape (n,) or (n, g)."""
        bids = np.asarray(bids, dtype=np.float64)
        one_d = bids.ndim == 1
        bids_2d = bids[:, None] if one_d else bids
        if bids_2d.shape[0] != len(X):
            raise ValueError(f"bids rows {bids_2d.shape[0]} != X rows {len(X)}")
        w = np.clip(self._win_prob_2d(X, segment, bids_2d), 0.0, 1.0)
        return w[:, 0] if one_d else w

    def predict(
        self, X: np.ndarray, segment: np.ndarray, bids: np.ndarray, chunk_size: int = 20_000
    ) -> np.ndarray:
        """Chunked win_prob to keep memory bounded on large test sets."""
        bids = np.asarray(bids, dtype=np.float64)
        out = np.empty(bids.shape, dtype=np.float64)
        for start in range(0, len(X), chunk_size):
            sl = slice(start, start + chunk_size)
            out[sl] = self.win_prob(X[sl], segment[sl], bids[sl])
        return out
