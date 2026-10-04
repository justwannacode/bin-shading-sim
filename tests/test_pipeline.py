import numpy as np
import pytest

pytest.importorskip("lightgbm")

from bidshade.config import Config  # noqa: E402
from bidshade.pipeline import run_experiment  # noqa: E402
from bidshade.synthetic import make_auction_data  # noqa: E402


def test_end_to_end_on_synthetic(tmp_path):
    cfg = Config.model_validate(
        {
            "kaplan_meier": {"segment_min_size": 300},
            "win_classifier": {"num_boost_round": 100, "early_stopping_rounds": 10},
            "hazard": {
                "num_boost_round": 80,
                "early_stopping_rounds": 10,
                "max_train_requests": 10_000,
            },
            "optimizer": {"grid_size": 40},
            "evaluation": {"win_curve_sample": 500},
            "output_dir": str(tmp_path / "run"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )
    train = make_auction_data(20_000, seed=0, day_offsets=(0, 5))
    valid = make_auction_data(5_000, seed=1, day_offsets=(5, 6))
    test = make_auction_data(5_000, seed=2, day_offsets=(6, 8))
    result = run_experiment(cfg, train, valid, test)

    rows = {r["strategy"]: r for r in result.strategies}
    assert rows["truthful"]["surplus_cny"] == 0.0
    assert np.isclose(rows["oracle"]["oracle_share"], 1.0)
    for name, row in rows.items():
        assert row["surplus_cny"] <= rows["oracle"]["surplus_cny"] + 1e-9, name
    assert (
        rows["shading_lgbm_win_classifier"]["surplus_cny"]
        > rows["shading_km_global"]["surplus_cny"]
    )
    # hindsight k is the best fixed k on test by construction
    assert rows["fixed_shading_hindsight"]["surplus_cny"] >= rows["fixed_shading"]["surplus_cny"]
    assert np.isclose(rows["fixed_shading_hindsight"]["uplift_vs_hindsight"], 0.0)
    for f in ("strategies.csv", "landscape_metrics.csv", "fixed_k_curve.csv", "config.yaml"):
        assert (tmp_path / "run" / f).exists()
    assert (tmp_path / "reports" / "results.md").exists()
    assert (tmp_path / "reports" / "figures" / "surplus_vs_k.png").exists()


def test_drop_features_ablation(tmp_path):
    cfg = Config.model_validate(
        {
            "kaplan_meier": {"enabled": False},
            "win_classifier": {"num_boost_round": 50, "early_stopping_rounds": 10},
            "hazard": {"enabled": False},
            "optimizer": {"grid_size": 20},
            "evaluation": {"win_curve_sample": 200},
            "drop_features": ["weekday", "hour"],
            "output_dir": str(tmp_path / "run"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )
    train = make_auction_data(5_000, seed=0, day_offsets=(0, 5))
    valid = make_auction_data(1_000, seed=1, day_offsets=(5, 6))
    test = make_auction_data(1_000, seed=2, day_offsets=(6, 8))
    run_experiment(cfg, train, valid, test)
    importance = (tmp_path / "run" / "feature_importance_lgbm_win_classifier.json").read_text()
    assert "weekday" not in importance and "hour" not in importance
    assert "Dropped features: weekday, hour" in (tmp_path / "reports" / "results.md").read_text()
