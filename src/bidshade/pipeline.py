"""End-to-end experiment: censored samples -> landscapes -> bid optimization -> simulation."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import plots
from .censoring import make_censored_sample
from .config import Config
from .dataset import AuctionData
from .landscape import GlobalKaplanMeier, Landscape, SegmentedKaplanMeier
from .metrics import anlp, brier_score, calibration_table, expected_calibration_error, log_loss
from .optimizer import make_q_grid, optimal_bids
from .simulator import CPM_FEN_TO_CNY, first_price_auction, summarize
from .strategies import (
    fixed_k_surplus_curve,
    fixed_shading_bids,
    oracle_bids,
    truthful_bids,
    tune_fixed_k,
)

log = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    strategies: list[dict]
    landscape_metrics: list[dict]
    best_k: float
    output_dir: Path


def build_landscapes(cfg: Config) -> list[Landscape]:
    models: list[Landscape] = []
    if cfg.kaplan_meier.enabled:
        models.append(GlobalKaplanMeier())
        if cfg.kaplan_meier.segmented:
            models.append(SegmentedKaplanMeier(min_size=cfg.kaplan_meier.segment_min_size))
    if cfg.win_classifier.enabled:
        from .landscape.win_classifier import LightGBMWinClassifier

        wc = cfg.win_classifier
        models.append(
            LightGBMWinClassifier(
                params=wc.params,
                num_boost_round=wc.num_boost_round,
                early_stopping_rounds=wc.early_stopping_rounds,
                monotone_method=wc.monotone_method,
            )
        )
    if cfg.hazard.enabled:
        from .landscape.hazard import LightGBMDiscreteHazard

        hz = cfg.hazard
        models.append(
            LightGBMDiscreteHazard(
                params=hz.params,
                bin_width=hz.bin_width,
                max_train_requests=hz.max_train_requests,
                max_valid_requests=hz.max_valid_requests,
                num_boost_round=hz.num_boost_round,
                early_stopping_rounds=hz.early_stopping_rounds,
                seed=cfg.censoring.seed,
            )
        )
    return models


def run_experiment(
    cfg: Config, train: AuctionData, valid: AuctionData, test: AuctionData
) -> ExperimentResult:
    out_dir = Path(cfg.output_dir)
    fig_dir = Path(cfg.reports_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    chunk = cfg.optimizer.chunk_size
    ev = cfg.evaluation
    if cfg.drop_features:
        log.info("dropping features: %s", cfg.drop_features)
        train, valid, test = (d.drop_features(cfg.drop_features) for d in (train, valid, test))

    # 1. Censored feedback for training and evaluation (different seeds per split).
    c = cfg.censoring
    train_cs = make_censored_sample(train, c.alpha_low, c.alpha_high, c.replicates, c.seed)
    valid_cs = make_censored_sample(valid, c.alpha_low, c.alpha_high, 1, c.seed + 1)
    test_cs = make_censored_sample(test, c.alpha_low, c.alpha_high, 1, c.seed + 2)
    for name, cs in (("train", train_cs), ("valid", valid_cs), ("test", test_cs)):
        log.info("censored %s: %d rows, win rate %.3f", name, len(cs), cs.win_rate)

    # 2. Fit landscapes.
    landscapes = build_landscapes(cfg)
    for model in landscapes:
        log.info("fitting %s", model.name)
        model.fit(train_cs, valid_cs)
        log.info("%s fitted in %.1fs", model.name, model.fit_seconds)

    # 3. Landscape quality on held-out test bids.
    landscape_rows, calib_tables = [], {}
    for model in landscapes:
        p = model.predict(test_cs.X, test_cs.segment, test_cs.bid, chunk_size=chunk)
        table = calibration_table(test_cs.won, p, ev.calibration_bins)
        calib_tables[model.name] = table
        landscape_rows.append(
            {
                "landscape": model.name,
                "log_loss": log_loss(test_cs.won, p),
                "brier": brier_score(test_cs.won, p),
                "ece": expected_calibration_error(table),
                "anlp": anlp(model, test, ev.anlp_bin_width, chunk),
                "fit_seconds": model.fit_seconds,
            }
        )
        log.info("landscape metrics: %s", landscape_rows[-1])

    # 4. Fixed shading baseline: tune k on validation logs.
    fs = cfg.fixed_shading
    k_grid = np.round(np.arange(fs.k_min, fs.k_max + 1e-9, fs.k_step), 6)
    best_k, valid_curve = tune_fixed_k(valid, k_grid)
    test_curve = fixed_k_surplus_curve(test, k_grid)
    # Unfair upper bound for the baseline: k picked with hindsight on the test set itself.
    hindsight_k = float(k_grid[int(np.argmax(test_curve))])
    log.info("best fixed k on valid: %.2f, hindsight-best k on test: %.2f", best_k, hindsight_k)

    # 5. Bids of every strategy on test.
    q_min = cfg.optimizer.q_min if cfg.optimizer.q_min is not None else c.alpha_low
    q_grid = make_q_grid(q_min, cfg.optimizer.grid_size)
    strategy_bids: dict[str, np.ndarray] = {
        "truthful": truthful_bids(test),
        "fixed_shading": fixed_shading_bids(test, best_k),
        "fixed_shading_hindsight": fixed_shading_bids(test, hindsight_k),
    }
    for model in landscapes:
        bids, _ = optimal_bids(model, test, q_grid, chunk_size=chunk)
        strategy_bids[f"shading_{model.name}"] = bids
    strategy_bids["oracle"] = oracle_bids(test, ev.oracle_eps)

    oracle_total = float(first_price_auction(test.v, test.z, strategy_bids["oracle"]).surplus.sum())
    strategy_rows = [
        summarize(n, test.v, test.z, b, oracle_total) for n, b in strategy_bids.items()
    ]
    by_name = {r["strategy"]: r for r in strategy_rows}
    fixed_surplus = by_name["fixed_shading"]["surplus_cny"]
    hindsight_surplus = by_name["fixed_shading_hindsight"]["surplus_cny"]
    for row in strategy_rows:
        row["uplift_vs_fixed"] = _uplift(row["surplus_cny"], fixed_surplus)
        row["uplift_vs_hindsight"] = _uplift(row["surplus_cny"], hindsight_surplus)
    by_name["fixed_shading"]["k"] = best_k
    by_name["fixed_shading_hindsight"]["k"] = hindsight_k

    # 6. Surplus by ad exchange.
    groups = [str(g) for g in np.unique(test.group)]
    group_rows, group_surplus = [], {}
    report_strategies = [
        s for s in strategy_bids if s not in ("truthful", "oracle", "fixed_shading_hindsight")
    ]
    for name in report_strategies:
        outcome = first_price_auction(test.v, test.z, strategy_bids[name])
        values = []
        for g in groups:
            mask = test.group.astype(str) == g
            surplus = float(outcome.surplus[mask].sum()) * CPM_FEN_TO_CNY
            values.append(surplus)
            group_rows.append(
                {
                    "strategy": name,
                    "group": g,
                    "requests": int(mask.sum()),
                    "win_rate": float(outcome.won[mask].mean()) if mask.any() else float("nan"),
                    "surplus_cny": surplus,
                }
            )
        group_surplus[name] = np.array(values)

    # 7. Save tables, report and figures.
    _write_csv(out_dir / "strategies.csv", strategy_rows)
    _write_csv(out_dir / "landscape_metrics.csv", landscape_rows)
    _write_csv(out_dir / "group_surplus.csv", group_rows)
    _write_csv(
        out_dir / "fixed_k_curve.csv",
        [
            {
                "k": k,
                "valid_surplus_cny": vs * CPM_FEN_TO_CNY,
                "test_surplus_cny": ts * CPM_FEN_TO_CNY,
            }
            for k, vs, ts in zip(k_grid, valid_curve, test_curve, strict=True)
        ],
    )
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(), f, allow_unicode=True, sort_keys=False)
    for model in landscapes:
        importance = getattr(model, "feature_importance", None)
        if callable(importance):
            ranked = dict(sorted(importance().items(), key=lambda kv: -kv[1]))
            (out_dir / f"feature_importance_{model.name}.json").write_text(
                json.dumps(ranked, indent=2), encoding="utf-8"
            )

    report_path = Path(cfg.reports_dir) / "results.md"
    report_path.write_text(
        _markdown_report(strategy_rows, landscape_rows, best_k, hindsight_k, len(test), cfg),
        encoding="utf-8",
    )

    model_names = [f"shading_{m.name}" for m in landscapes]
    plots.plot_fixed_k_curve(
        k_grid,
        test_curve * CPM_FEN_TO_CNY,
        best_k,
        hindsight_k,
        {n: r["surplus_cny"] for r in strategy_rows if (n := r["strategy"]) in model_names},
        oracle_total * CPM_FEN_TO_CNY,
        fig_dir / "surplus_vs_k.png",
    )
    plots.plot_calibration(calib_tables, fig_dir / "calibration.png")
    plots.plot_group_surplus(groups, group_surplus, fig_dir / "surplus_by_exchange.png")
    plots.plot_bid_ratio_hist(
        {
            n: strategy_bids[n][strategy_bids[n] > 0] / test.v[strategy_bids[n] > 0]
            for n in model_names
        },
        fig_dir / "bid_to_value.png",
    )
    _plot_average_landscape(
        landscapes, test, ev.win_curve_sample, chunk, fig_dir / "win_curves.png"
    )

    if cfg.mlflow.enabled:
        _log_mlflow(cfg, strategy_rows, landscape_rows, best_k, out_dir, fig_dir)

    return ExperimentResult(strategy_rows, landscape_rows, best_k, out_dir)


def _uplift(value: float, base: float) -> float:
    return value / base - 1.0 if base > 0 else float("nan")


def _plot_average_landscape(
    landscapes: list[Landscape], test: AuctionData, n_sample: int, chunk: int, path: Path
) -> None:
    sample = test.sample(n_sample, seed=0)
    grid = np.linspace(0.0, float(np.max(sample.v)), 61)
    empirical = (sample.z[:, None] < grid[None, :]).mean(axis=0)
    bids = np.tile(grid, (len(sample), 1))
    curves = {
        m.name: m.predict(sample.X, sample.segment, bids, chunk_size=chunk).mean(axis=0)
        for m in landscapes
    }
    plots.plot_win_curves(grid, empirical, curves, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x: float, digits: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{digits}f}"


def _markdown_report(
    strategy_rows: list[dict],
    landscape_rows: list[dict],
    best_k: float,
    hindsight_k: float,
    n_test: int,
    cfg: Config,
) -> str:
    dropped = ", ".join(cfg.drop_features) if cfg.drop_features else "none"
    lines = [
        "# Results",
        "",
        f"Test requests: {n_test:,}. Fixed shading k tuned on validation: {best_k:.2f}; "
        f"hindsight-best k on test: {hindsight_k:.2f}. Dropped features: {dropped}.",
        "",
        "`fixed_shading_hindsight` picks k on the test set itself: an unfair upper bound "
        "for the fixed baseline.",
        "",
        "## Strategies (first-price simulation on test)",
        "",
        "| Strategy | Surplus, CNY | vs fixed | vs hindsight fixed | Win rate | Spend, CNY | "
        "Avg paid CPM, fen | Share of oracle |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in strategy_rows:
        lines.append(
            f"| {r['strategy']} | {_fmt(r['surplus_cny'])} | "
            f"{_fmt(r['uplift_vs_fixed'] * 100, 1)}% | {_fmt(r['uplift_vs_hindsight'] * 100, 1)}% | "
            f"{_fmt(r['win_rate'] * 100, 1)}% | "
            f"{_fmt(r['spend_cny'])} | {_fmt(r['avg_paid_cpm_fen'], 1)} | "
            f"{_fmt(r['oracle_share'] * 100, 1)}% |"
        )
    lines += [
        "",
        "## Landscape quality (test)",
        "",
        "| Landscape | Log loss | Brier | ECE | ANLP | Fit, s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in landscape_rows:
        lines.append(
            f"| {r['landscape']} | {_fmt(r['log_loss'], 4)} | {_fmt(r['brier'], 4)} | "
            f"{_fmt(r['ece'], 4)} | {_fmt(r['anlp'], 3)} | {_fmt(r['fit_seconds'], 1)} |"
        )
    return "\n".join(lines) + "\n"


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = str(value)[:250]
    return flat


def _log_mlflow(
    cfg: Config,
    strategy_rows: list[dict],
    landscape_rows: list[dict],
    best_k: float,
    out_dir: Path,
    fig_dir: Path,
) -> None:
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow is not installed: `uv sync --extra tracking` to enable tracking")
        return

    def clean(name: str) -> str:
        return re.sub(r"[^\w\-./ ]", "_", name)

    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_params(_flatten(cfg.model_dump()))
        mlflow.log_metric("fixed_shading.best_k", best_k)
        for row in strategy_rows:
            for key in (
                "surplus_cny",
                "win_rate",
                "spend_cny",
                "oracle_share",
                "uplift_vs_fixed",
                "uplift_vs_hindsight",
            ):
                value = row.get(key)
                if value is not None and np.isfinite(value):
                    mlflow.log_metric(clean(f"{row['strategy']}.{key}"), float(value))
        for row in landscape_rows:
            for key in ("log_loss", "brier", "ece", "anlp", "fit_seconds"):
                mlflow.log_metric(clean(f"{row['landscape']}.{key}"), float(row[key]))
        mlflow.log_artifacts(str(out_dir), artifact_path="run")
        mlflow.log_artifacts(str(fig_dir), artifact_path="figures")
