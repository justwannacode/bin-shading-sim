"""Non-parametric landscape baselines: Kaplan-Meier on right-censored market prices.

The estimator is implemented in numpy (fast on millions of rows); tests cross-check it
against lifelines.KaplanMeierFitter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..censoring import CensoredSample
from .base import Landscape


@dataclass
class KMCurve:
    event_times: np.ndarray  # sorted unique prices with observed events
    survival: np.ndarray  # S(t_k) = P(z > t_k)

    @classmethod
    def fit(cls, durations: np.ndarray, events: np.ndarray) -> KMCurve:
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=bool)
        if not events.any():
            return cls(np.empty(0), np.empty(0))
        sorted_durations = np.sort(durations)
        times, deaths = np.unique(durations[events], return_counts=True)
        at_risk = len(durations) - np.searchsorted(sorted_durations, times, side="left")
        survival = np.cumprod(1.0 - deaths / at_risk)
        return cls(times, survival)

    def win_prob(self, bids: np.ndarray) -> np.ndarray:
        """W(b) = P(z < b) = 1 - S(b-), where S(b-) is S at the last event time < b."""
        bids = np.asarray(bids, dtype=np.float64)
        if self.event_times.size == 0:
            return np.zeros_like(bids)
        idx = np.searchsorted(self.event_times, bids, side="left") - 1
        surv_before = np.where(idx >= 0, self.survival[np.clip(idx, 0, None)], 1.0)
        return 1.0 - surv_before


class GlobalKaplanMeier(Landscape):
    """One landscape for the whole market: ignores features."""

    name = "km_global"

    def fit(self, train: CensoredSample, valid: CensoredSample | None = None) -> GlobalKaplanMeier:
        start = time.perf_counter()
        self.curve = KMCurve.fit(train.durations, train.won)
        self.fit_seconds = time.perf_counter() - start
        return self

    def _win_prob_2d(self, X: np.ndarray, segment: np.ndarray, bids: np.ndarray) -> np.ndarray:
        return self.curve.win_prob(bids)


class SegmentedKaplanMeier(Landscape):
    """Separate KM curve per segment (exchange x slot size x hour bucket).

    Segments with fewer than `min_size` training rows, and the "rare" code 0,
    fall back to the global curve.
    """

    name = "km_segmented"

    def __init__(self, min_size: int = 2000) -> None:
        self.min_size = min_size

    def fit(
        self, train: CensoredSample, valid: CensoredSample | None = None
    ) -> SegmentedKaplanMeier:
        start = time.perf_counter()
        durations, events = train.durations, train.won
        self.global_curve = KMCurve.fit(durations, events)
        self.curves: dict[int, KMCurve] = {}

        order = np.argsort(train.segment, kind="stable")
        segments, starts, counts = np.unique(
            train.segment[order], return_index=True, return_counts=True
        )
        for seg, st, cnt in zip(segments, starts, counts, strict=True):
            if seg == 0 or cnt < self.min_size:
                continue
            idx = order[st : st + cnt]
            self.curves[int(seg)] = KMCurve.fit(durations[idx], events[idx])
        self.fit_seconds = time.perf_counter() - start
        return self

    @property
    def n_segments(self) -> int:
        return len(self.curves)

    def _win_prob_2d(self, X: np.ndarray, segment: np.ndarray, bids: np.ndarray) -> np.ndarray:
        out = np.empty(bids.shape, dtype=np.float64)
        for seg in np.unique(segment):
            rows = segment == seg
            curve = self.curves.get(int(seg), self.global_curve)
            out[rows] = curve.win_prob(bids[rows])
        return out
