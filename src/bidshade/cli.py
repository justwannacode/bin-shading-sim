"""Command line entry point: `bidshade {synthetic,prepare,eda,run}`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .config import load_config

log = logging.getLogger("bidshade")


def _cmd_synthetic(args: argparse.Namespace) -> None:
    from .synthetic import write_make_ipinyou_format

    path = write_make_ipinyou_format(
        args.out, args.advertiser, args.n_train, args.n_test, args.seed
    )
    log.info("synthetic logs written to %s", path)


def _cmd_prepare(args: argparse.Namespace) -> None:
    from .data import prepare

    cfg = load_config(args.config)
    prepare(cfg.data)


def _load_splits(processed_dir: str):
    from .data import load_split

    return tuple(load_split(processed_dir, name) for name in ("train", "valid", "test"))


def _cmd_eda(args: argparse.Namespace) -> None:
    from .plots import plot_price_eda

    cfg = load_config(args.config)
    splits = dict(
        zip(("train", "valid", "test"), _load_splits(cfg.data.processed_dir), strict=True)
    )
    for name, d in splits.items():
        ratio = d.z / d.v
        print(
            f"{name:>5}: {len(d):>9,} rows | v: median {np.median(d.v):6.1f}, "
            f"unique {len(np.unique(d.v)):>4} | z: median {np.median(d.z):6.1f}, "
            f"p90 {np.percentile(d.z, 90):6.1f} | z/v median {np.median(ratio):.3f}"
        )
    train = splits["train"]
    print("\nmarket price by ad exchange (train):")
    for g in np.unique(train.group):
        z = train.z[train.group == g]
        print(
            f"  exchange {g:>4}: {len(z):>9,} rows, median z {np.median(z):6.1f}, mean {z.mean():6.1f}"
        )
    _print_daily_prices(cfg.data.processed_dir)
    path = plot_price_eda(
        train.z, train.v, train.group, Path(cfg.reports_dir) / "figures" / "eda.png"
    )
    print(f"\nfigure: {path}")


def _print_daily_prices(processed_dir: str) -> None:
    """Market price by day and advertiser: separates real price drift from campaign mix shifts."""
    import json

    import polars as pl

    base = Path(processed_dir)
    vocab = json.loads((base / "vocabularies.json").read_text(encoding="utf-8"))["advertiser"]
    names = {i + 1: name for i, name in enumerate(vocab)}
    frames = [
        pl.read_parquet(base / f"{split}.parquet").with_columns(pl.lit(split).alias("split"))
        for split in ("train", "valid", "test")
    ]
    daily = (
        pl.concat(frames)
        .group_by("day", "split", "advertiser")
        .agg(pl.len().alias("rows"), pl.col("z").median().alias("median_z"))
        .sort("day", "advertiser")
    )
    print("\nmarket price by day and advertiser:")
    for r in daily.iter_rows(named=True):
        adv = names.get(r["advertiser"], "other")
        print(
            f"  {r['day']} {r['split']:>5} adv {adv:>5}: {r['rows']:>9,} rows, median z {r['median_z']:6.1f}"
        )


def _cmd_run(args: argparse.Namespace) -> None:
    from .pipeline import run_experiment

    cfg = load_config(args.config)
    if args.drop_features:
        cfg.drop_features = [f.strip() for f in args.drop_features.split(",") if f.strip()]
    if args.tag:
        cfg.output_dir = f"{cfg.output_dir}_{args.tag}"
        cfg.reports_dir = str(Path(cfg.reports_dir) / args.tag)
    train, valid, test = _load_splits(cfg.data.processed_dir)
    result = run_experiment(cfg, train, valid, test)

    print(f"\nFixed shading k (tuned on valid): {result.best_k:.2f}\n")
    print(
        f"{'strategy':<34}{'surplus, CNY':>14}{'vs fixed':>10}{'vs hindsight':>14}"
        f"{'win rate':>10}{'oracle %':>10}"
    )
    for r in result.strategies:
        print(
            f"{r['strategy']:<34}{r['surplus_cny']:>14,.2f}{r['uplift_vs_fixed'] * 100:>9.1f}%"
            f"{r['uplift_vs_hindsight'] * 100:>13.1f}%"
            f"{r['win_rate'] * 100:>9.1f}%{r['oracle_share'] * 100:>9.1f}%"
        )
    print(f"\nTables: {result.output_dir}  Report: {Path(cfg.reports_dir) / 'results.md'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bidshade", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthetic", help="write synthetic logs in make-ipinyou-data format")
    syn.add_argument("--out", default="data/synthetic")
    syn.add_argument("--advertiser", default="9999")
    syn.add_argument("--n-train", type=int, default=200_000)
    syn.add_argument("--n-test", type=int, default=80_000)
    syn.add_argument("--seed", type=int, default=0)
    syn.set_defaults(func=_cmd_synthetic)

    for name, func, help_text in (
        ("prepare", _cmd_prepare, "raw logs -> features -> parquet splits"),
        ("eda", _cmd_eda, "quick statistics and a price distribution figure"),
        ("run", _cmd_run, "fit landscapes, optimize bids, simulate, write the report"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default="configs/default.yaml")
        if name == "run":
            p.add_argument("--drop-features", default="", help="comma-separated, e.g. weekday")
            p.add_argument("--tag", default="", help="suffix for output dirs of this run")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args.func(args)


if __name__ == "__main__":
    main()
