"""Figures for the report (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
# Same color for the same model in every figure.
NAMED_COLORS = {
    "fixed_shading": "#7f7f7f",
    "km_global": "#1f77b4",
    "km_segmented": "#ff7f0e",
    "lgbm_win_classifier": "#2ca02c",
    "lgbm_discrete_hazard": "#d62728",
}


def _color(name: str, i: int) -> str:
    return NAMED_COLORS.get(name.removeprefix("shading_"), PALETTE[i % len(PALETTE)])


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_fixed_k_curve(
    k_grid: np.ndarray,
    test_curve_cny: np.ndarray,
    best_k: float,
    hindsight_k: float,
    model_surplus_cny: dict[str, float],
    oracle_surplus_cny: float,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_grid, test_curve_cny, color="black", lw=2, label="fixed shading b = k·v (test)")
    ax.axvline(best_k, color="gray", ls=":", label=f"k tuned on valid = {best_k:.2f}")
    ax.axvline(
        hindsight_k,
        color="gray",
        ls="--",
        alpha=0.6,
        label=f"hindsight-best k on test = {hindsight_k:.2f}",
    )
    for i, (name, value) in enumerate(model_surplus_cny.items()):
        ax.axhline(value, color=_color(name, i), ls="--", label=name)
    ax.axhline(oracle_surplus_cny, color="black", ls="-.", alpha=0.4, label="oracle")
    ax.set_ylim(bottom=0.5 * float(np.max(test_curve_cny)))  # zoom in on the region that matters
    ax.set_xlabel("shading factor k")
    ax.set_ylabel("total surplus on test, CNY")
    ax.set_title("Surplus: fixed shading vs landscape-based shading")
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_calibration(tables: dict[str, dict[str, np.ndarray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    for i, (name, t) in enumerate(tables.items()):
        ax.plot(t["mean_pred"], t["frac_won"], marker="o", ms=4, color=_color(name, i), label=name)
    ax.set_xlabel("predicted win probability W(b' | x)")
    ax.set_ylabel("observed win rate")
    ax.set_title("Calibration on test bids")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_win_curves(
    bid_grid: np.ndarray,
    empirical: np.ndarray,
    model_curves: dict[str, np.ndarray],
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bid_grid, empirical, color="black", lw=2.5, label="empirical P(z < b)")
    for i, (name, curve) in enumerate(model_curves.items()):
        ax.plot(bid_grid, curve, color=_color(name, i), ls="--", label=name)
    ax.set_xlabel("bid b, CPM fen")
    ax.set_ylabel("average win probability")
    ax.set_title("Average bid landscape on test")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_group_surplus(
    groups: list[str], surplus_by_strategy: dict[str, np.ndarray], path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = len(surplus_by_strategy)
    width = 0.8 / max(n, 1)
    x = np.arange(len(groups))
    for i, (name, values) in enumerate(surplus_by_strategy.items()):
        ax.bar(x + (i - (n - 1) / 2) * width, values, width, color=_color(name, i), label=name)
    ax.set_xticks(x, groups)
    ax.set_xlabel("ad exchange")
    ax.set_ylabel("surplus on test, CNY")
    ax.set_title("Surplus by ad exchange")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def plot_bid_ratio_hist(ratios: dict[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 41)
    for i, (name, r) in enumerate(ratios.items()):
        ax.hist(
            r, bins=bins, histtype="step", lw=1.8, density=True, color=_color(name, i), label=name
        )
    ax.set_xlabel("bid / value")
    ax.set_ylabel("density")
    ax.set_title("How much each model shades")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_price_eda(z: np.ndarray, v: np.ndarray, group: np.ndarray, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, float(np.percentile(v, 99.5)), 61)
    for i, g in enumerate(np.unique(group)):
        axes[0].hist(
            z[group == g],
            bins=bins,
            histtype="step",
            density=True,
            lw=1.5,
            color=PALETTE[i % len(PALETTE)],
            label=f"exchange {g}",
        )
    axes[0].set_xlabel("market price z (payprice), CPM fen")
    axes[0].set_title("Market price by exchange")
    axes[0].legend(fontsize=8)
    axes[1].hist(z / v, bins=np.linspace(0, 1, 51), color=PALETTE[0])
    axes[1].set_xlabel("z / v")
    axes[1].set_title("Market price relative to value")
    for ax in axes:
        ax.grid(alpha=0.3)
    return _save(fig, path)
