"""In-memory auction data used by models and the simulator (numpy only, no polars)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Features known at bid time. Categorical ones are integer codes (0 = rare / unknown).
CATEGORICAL_FEATURES: list[str] = [
    "region",
    "city",
    "adexchange",
    "domain",
    "slot_size",
    "slotvisibility",
    "slotformat",
    "ua_os",
    "ua_browser",
    "advertiser",
]
NUMERIC_FEATURES: list[str] = [
    "hour",
    "weekday",
    "slotwidth",
    "slotheight",
    "slot_area",
    "slotprice",
]
FEATURE_NAMES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class AuctionData:
    """One row per bid request that the original bidder won in the logs.

    Prices are in iPinYou units: fen (0.01 CNY) per 1000 impressions (CPM).
    v: value of the impression for the advertiser (bidprice from the log).
    z: highest competing price (payprice from the second-price log).
    """

    X: np.ndarray
    v: np.ndarray
    z: np.ndarray
    segment: np.ndarray
    group: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    categorical_features: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))

    def __post_init__(self) -> None:
        n = len(self.v)
        for name in ("X", "z", "segment", "group"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"AuctionData.{name} has {len(getattr(self, name))} rows, v has {n}"
                )
        if self.X.ndim != 2 or self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X shape {self.X.shape} does not match {len(self.feature_names)} features"
            )

    def __len__(self) -> int:
        return len(self.v)

    def take(self, idx: np.ndarray) -> AuctionData:
        return AuctionData(
            X=self.X[idx],
            v=self.v[idx],
            z=self.z[idx],
            segment=self.segment[idx],
            group=self.group[idx],
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
        )

    def drop_features(self, names: list[str]) -> AuctionData:
        unknown = sorted(set(names) - set(self.feature_names))
        if unknown:
            raise ValueError(
                f"Unknown features to drop: {unknown}; available: {self.feature_names}"
            )
        keep = [i for i, f in enumerate(self.feature_names) if f not in names]
        return AuctionData(
            X=self.X[:, keep],
            v=self.v,
            z=self.z,
            segment=self.segment,
            group=self.group,
            feature_names=[self.feature_names[i] for i in keep],
            categorical_features=[f for f in self.categorical_features if f not in names],
        )

    def sample(self, n_max: int | None, seed: int = 0) -> AuctionData:
        if n_max is None or len(self) <= n_max:
            return self
        idx = np.sort(np.random.default_rng(seed).choice(len(self), size=n_max, replace=False))
        return self.take(idx)
