"""Main model: LightGBM classifier of P(win | x, b) with a monotone constraint on the bid.

Each censored row is a binary outcome at a known bid, so censoring is handled for free:
the model only needs win/loss labels. That is also the only feedback a first-price
bidder is guaranteed to get, which makes this model the most realistic of the set.
"""

from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np

from ..censoring import CensoredSample
from .base import Landscape

log = logging.getLogger(__name__)

BID_FEATURE = "bid"


class LightGBMWinClassifier(Landscape):
    name = "lgbm_win_classifier"

    def __init__(
        self,
        params: dict,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50,
        monotone_method: str = "advanced",
        log_period: int = 100,
    ) -> None:
        self.params = dict(params)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.monotone_method = monotone_method
        self.log_period = log_period
        self.booster: lgb.Booster | None = None
        self.best_iteration: int | None = None

    @staticmethod
    def _design(X: np.ndarray, bids: np.ndarray) -> np.ndarray:
        return np.column_stack([X, bids]).astype(np.float32, copy=False)

    def fit(
        self, train: CensoredSample, valid: CensoredSample | None = None
    ) -> LightGBMWinClassifier:
        start = time.perf_counter()
        self.feature_names = [*train.feature_names, BID_FEATURE]
        categorical = list(train.categorical_features)
        n_base = len(train.feature_names)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            **self.params,
            "monotone_constraints": [0] * n_base + [1],
            "monotone_constraints_method": self.monotone_method,
        }

        dtrain = lgb.Dataset(
            self._design(train.X, train.bid),
            label=train.won.astype(np.float32),
            feature_name=self.feature_names,
            categorical_feature=categorical,
            free_raw_data=True,
        )
        callbacks = [lgb.log_evaluation(period=self.log_period)]
        extra: dict = {}
        if valid is not None and len(valid):
            dvalid = lgb.Dataset(
                self._design(valid.X, valid.bid),
                label=valid.won.astype(np.float32),
                feature_name=self.feature_names,
                categorical_feature=categorical,
                reference=dtrain,
            )
            extra = {"valid_sets": [dvalid], "valid_names": ["valid"]}
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            callbacks=callbacks,
            **extra,
        )
        self.best_iteration = self.booster.best_iteration or self.booster.current_iteration()
        self.fit_seconds = time.perf_counter() - start
        log.info("%s: %d trees, %.1fs", self.name, self.best_iteration, self.fit_seconds)
        return self

    def _win_prob_2d(self, X: np.ndarray, segment: np.ndarray, bids: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted")
        n, g = bids.shape
        design = self._design(np.repeat(X, g, axis=0), bids.reshape(-1))
        p = self.booster.predict(design, num_iteration=self.best_iteration)
        return np.asarray(p, dtype=np.float64).reshape(n, g)

    def feature_importance(self) -> dict[str, float]:
        if self.booster is None:
            return {}
        gain = self.booster.feature_importance(importance_type="gain")
        return dict(zip(self.feature_names, map(float, gain), strict=True))
