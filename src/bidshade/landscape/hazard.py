"""Discrete-time hazard model of the market price on LightGBM.

Prices are cut into bins [k*w, (k+1)*w). The model learns the hazard
    h_k(x) = P(z in bin k | z >= k*w, x)
from a "person-period" table: each request is expanded into one row per bin it survived.
  * win with observed z in bin k_z: rows for bins 0..k_z, label 1 only at k_z;
  * loss at bid b': rows for bins fully below b' (upper edge <= b'), all labels 0.
Then P(z >= (k+1)*w | x) = prod_{j<=k} (1 - h_j(x)) and W(b) is linearly interpolated
between bin edges. Unlike the win classifier, this gives the full distribution of z.
"""

from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np

from ..censoring import CensoredSample
from .base import Landscape

log = logging.getLogger(__name__)

BIN_FEATURE = "price_bin_lower"


class LightGBMDiscreteHazard(Landscape):
    name = "lgbm_discrete_hazard"

    def __init__(
        self,
        params: dict,
        bin_width: float = 5.0,
        max_train_requests: int | None = 300_000,
        max_valid_requests: int | None = 100_000,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50,
        log_period: int = 100,
        seed: int = 0,
    ) -> None:
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")
        self.params = dict(params)
        self.bin_width = float(bin_width)
        self.max_train_requests = max_train_requests
        self.max_valid_requests = max_valid_requests
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.log_period = log_period
        self.seed = seed
        self.booster: lgb.Booster | None = None

    def setup_bins(self, max_price: float) -> None:
        self.n_bins = int(np.ceil(max_price / self.bin_width)) + 1
        self.edges = np.arange(self.n_bins + 1, dtype=np.float64) * self.bin_width

    def expand(self, sample: CensoredSample) -> tuple[np.ndarray, np.ndarray]:
        """Person-period expansion. Returns (design matrix, binary labels)."""
        w, k_max = self.bin_width, self.n_bins - 1
        z = np.where(sample.won, sample.price_obs, 0.0)
        k_event = np.clip(np.floor(z / w).astype(np.int64), 0, k_max)
        k_censor = np.clip(np.floor(sample.bid / w).astype(np.int64), 0, self.n_bins)
        counts = np.where(sample.won, k_event + 1, k_censor)

        rows = np.repeat(np.arange(len(sample)), counts)
        starts = np.cumsum(counts) - counts
        k = np.arange(counts.sum()) - np.repeat(starts, counts)
        labels = sample.won[rows] & (k == k_event[rows])
        design = np.column_stack([sample.X[rows], self.edges[k]]).astype(np.float32, copy=False)
        return design, labels

    def fit(
        self, train: CensoredSample, valid: CensoredSample | None = None
    ) -> LightGBMDiscreteHazard:
        start = time.perf_counter()
        self.setup_bins(float(np.max(train.v)))
        self.feature_names = [*train.feature_names, BIN_FEATURE]
        categorical = list(train.categorical_features)
        params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, **self.params}

        X_tr, y_tr = self.expand(train.sample(self.max_train_requests, self.seed))
        log.info("%s: %d person-period rows for training", self.name, len(y_tr))
        dtrain = lgb.Dataset(
            X_tr,
            label=y_tr.astype(np.float32),
            feature_name=self.feature_names,
            categorical_feature=categorical,
            free_raw_data=True,
        )
        callbacks = [lgb.log_evaluation(period=self.log_period)]
        extra: dict = {}
        if valid is not None and len(valid):
            X_va, y_va = self.expand(valid.sample(self.max_valid_requests, self.seed + 1))
            dvalid = lgb.Dataset(
                X_va,
                label=y_va.astype(np.float32),
                feature_name=self.feature_names,
                categorical_feature=categorical,
                reference=dtrain,
            )
            extra = {"valid_sets": [dvalid], "valid_names": ["valid"]}
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            params, dtrain, num_boost_round=self.num_boost_round, callbacks=callbacks, **extra
        )
        self.best_iteration = self.booster.best_iteration or self.booster.current_iteration()
        self.fit_seconds = time.perf_counter() - start
        log.info("%s: %d trees, %.1fs", self.name, self.best_iteration, self.fit_seconds)
        return self

    def hazards(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted")
        n, k = len(X), self.n_bins
        design = np.column_stack([np.repeat(X, k, axis=0), np.tile(self.edges[:-1], n)])
        h = self.booster.predict(design.astype(np.float32), num_iteration=self.best_iteration)
        return np.asarray(h, dtype=np.float64).reshape(n, k)

    def cdf_at_edges(self, X: np.ndarray) -> np.ndarray:
        """W at bin edges, shape (n, n_bins + 1); W(0) = 0."""
        survival = np.cumprod(1.0 - self.hazards(X), axis=1)
        return np.concatenate([np.zeros((len(X), 1)), 1.0 - survival], axis=1)

    def _win_prob_2d(self, X: np.ndarray, segment: np.ndarray, bids: np.ndarray) -> np.ndarray:
        w_edges = self.cdf_at_edges(X)
        pos = bids / self.bin_width
        k = np.clip(np.floor(pos).astype(np.int64), 0, self.n_bins - 1)
        frac = np.clip(pos - k, 0.0, 1.0)  # bids beyond the last edge get W at the last edge
        lo = np.take_along_axis(w_edges, k, axis=1)
        hi = np.take_along_axis(w_edges, k + 1, axis=1)
        return lo + frac * (hi - lo)
