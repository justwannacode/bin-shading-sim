"""Experiment configuration: YAML file -> validated pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


def _default_lgbm_params() -> dict:
    return {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_cat_to_onehot": 4,
        "cat_smooth": 10.0,
        "seed": 42,
    }


class DataConfig(_Strict):
    # "make_ipinyou": <raw_dir>/<advertiser>/{train,test}.log.txt from wnzhang/make-ipinyou-data
    # "raw": original iPinYou impression logs (<raw_dir>/<raw_glob>), split by day here
    source: Literal["make_ipinyou", "raw"] = "make_ipinyou"
    raw_dir: str = "data/ipinyou"
    raw_glob: str = "training2nd/imp.*.txt"
    advertisers: list[str] = Field(default_factory=lambda: ["1458", "3386"])
    processed_dir: str = "artifacts/data"
    valid_days: int = 1
    test_days: int = 2  # used only for source="raw"; make_ipinyou has its own test file
    min_category_count: int = 20
    max_domains: int = 1000
    max_train_rows: int | None = None
    max_eval_rows: int | None = None
    seed: int = 42


class CensoringConfig(_Strict):
    # Test bid b' = alpha * v, alpha ~ U(alpha_low, alpha_high)
    alpha_low: float = 0.05
    alpha_high: float = 1.0
    replicates: int = 1
    seed: int = 7


class KaplanMeierConfig(_Strict):
    enabled: bool = True
    segmented: bool = True
    segment_min_size: int = 2000


class WinClassifierConfig(_Strict):
    enabled: bool = True
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50
    monotone_method: Literal["basic", "intermediate", "advanced"] = "advanced"
    params: dict = Field(default_factory=_default_lgbm_params)


class HazardConfig(_Strict):
    enabled: bool = True
    bin_width: float = 5.0
    max_train_requests: int = 300_000
    max_valid_requests: int = 100_000
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50
    params: dict = Field(default_factory=_default_lgbm_params)


class OptimizerConfig(_Strict):
    grid_size: int = 96
    q_min: float | None = None  # lowest bid as a share of v; None -> censoring.alpha_low
    chunk_size: int = 20_000


class FixedShadingConfig(_Strict):
    k_min: float = 0.05
    k_max: float = 1.0
    k_step: float = 0.01


class EvaluationConfig(_Strict):
    anlp_bin_width: float = 5.0
    calibration_bins: int = 15
    oracle_eps: float = 0.01
    win_curve_sample: int = 5000


class MlflowConfig(_Strict):
    enabled: bool = False
    experiment: str = "bid-shading"
    tracking_uri: str | None = None
    run_name: str | None = None


class Config(_Strict):
    data: DataConfig = Field(default_factory=DataConfig)
    censoring: CensoringConfig = Field(default_factory=CensoringConfig)
    kaplan_meier: KaplanMeierConfig = Field(default_factory=KaplanMeierConfig)
    win_classifier: WinClassifierConfig = Field(default_factory=WinClassifierConfig)
    hazard: HazardConfig = Field(default_factory=HazardConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    fixed_shading: FixedShadingConfig = Field(default_factory=FixedShadingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    drop_features: list[str] = Field(default_factory=list)  # ablations, e.g. ["weekday"]
    output_dir: str = "artifacts/run"
    reports_dir: str = "reports"


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
