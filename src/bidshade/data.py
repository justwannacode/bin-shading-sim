"""Load iPinYou logs, build bid-time features, split by day, save parquet (polars)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from .config import DataConfig
from .dataset import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES, AuctionData
from .schema import MAKE_IPINYOU_COLUMNS, RAW_IMP_COLUMNS

log = logging.getLogger(__name__)

NEEDED_COLUMNS = [
    "timestamp", "useragent", "region", "city", "adexchange", "domain", "slotwidth",
    "slotheight", "slotvisibility", "slotformat", "slotprice", "bidprice", "payprice",
    "advertiser",
]  # fmt: skip
_NUMERIC_RAW = ["slotwidth", "slotheight", "slotprice", "bidprice", "payprice"]
_STRING_RAW = [
    "region",
    "city",
    "adexchange",
    "domain",
    "slotvisibility",
    "slotformat",
    "advertiser",
]
_SEGMENT = "segment_key"


# ----------------------------------------------------------------------------- reading


def _has_header(path: Path) -> bool:
    with open(path, encoding="utf-8", errors="replace") as f:
        return "payprice" in f.readline().lower()


def _scan_tsv(path: Path, default_columns: list[str]) -> pl.LazyFrame:
    """All columns as strings: robust to dirty values; casting happens later."""
    has_header = _has_header(path)
    lf = pl.scan_csv(
        path,
        separator="\t",
        has_header=has_header,
        new_columns=None if has_header else default_columns,
        infer_schema_length=0,  # read every column as String
        quote_char=None,
        truncate_ragged_lines=True,
        encoding="utf8-lossy",
    )
    names = lf.collect_schema().names()
    return lf.rename({c: c.lower() for c in names if c != c.lower()})


def _select_needed(lf: pl.LazyFrame, path: Path, advertiser: str | None = None) -> pl.LazyFrame:
    names = lf.collect_schema().names()
    if "advertiser" not in names and advertiser is not None:
        lf = lf.with_columns(pl.lit(advertiser).alias("advertiser"))
        names = [*names, "advertiser"]
    missing = [c for c in NEEDED_COLUMNS if c not in names]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {names}")
    return lf.select(NEEDED_COLUMNS)


def load_make_ipinyou(cfg: DataConfig) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """<raw_dir>/<advertiser>/{train,test}.log.txt produced by wnzhang/make-ipinyou-data."""
    trains, tests = [], []
    for adv in cfg.advertisers:
        for split, bucket in (("train", trains), ("test", tests)):
            path = Path(cfg.raw_dir) / adv / f"{split}.log.txt"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Expected make-ipinyou-data output: "
                    f"<raw_dir>/<advertiser>/train.log.txt and test.log.txt"
                )
            bucket.append(_select_needed(_scan_tsv(path, MAKE_IPINYOU_COLUMNS), path, adv))
    return pl.concat(trains), pl.concat(tests)


def load_raw(cfg: DataConfig) -> pl.LazyFrame:
    """Original iPinYou impression logs, e.g. training2nd/imp.20130606.txt (decompressed)."""
    paths = sorted(Path(cfg.raw_dir).glob(cfg.raw_glob))
    if not paths:
        compressed = sorted(Path(cfg.raw_dir).glob(cfg.raw_glob + ".bz2"))
        hint = (
            f" Found {len(compressed)} .bz2 files: decompress them first, e.g. "
            f"`bzip2 -dk {Path(cfg.raw_dir) / cfg.raw_glob}.bz2`"
            if compressed
            else ""
        )
        raise FileNotFoundError(f"No files match {Path(cfg.raw_dir) / cfg.raw_glob}.{hint}")
    log.info("reading %d raw impression files", len(paths))
    lf = pl.concat([_select_needed(_scan_tsv(p, RAW_IMP_COLUMNS), p) for p in paths])
    return lf.filter(pl.col("advertiser").str.strip_chars().is_in(cfg.advertisers))


# ----------------------------------------------------------------------------- features


def _ua_os(ua: pl.Expr) -> pl.Expr:
    return (
        pl.when(ua.str.contains("android")).then(pl.lit("android"))
        .when(ua.str.contains("iphone|ipad|ipod|ios")).then(pl.lit("ios"))
        .when(ua.str.contains("windows")).then(pl.lit("windows"))
        .when(ua.str.contains("mac")).then(pl.lit("mac"))
        .when(ua.str.contains("linux")).then(pl.lit("linux"))
        .otherwise(pl.lit("other"))
    )  # fmt: skip


def _ua_browser(ua: pl.Expr) -> pl.Expr:
    # Order matters: Chinese browsers embed "chrome"/"msie" tokens in their UA strings.
    rules = [
        ("maxthon", "maxthon"),
        ("sogou|metasr", "sogou"),
        ("theworld", "theworld"),
        ("360se|360ee|qihoo", "360"),
        ("qqbrowser|tencenttraveler", "qq"),
        ("opera|opr/", "opera"),
        ("firefox", "firefox"),
        ("chrome|crios", "chrome"),
        ("safari", "safari"),
        ("msie|trident|(^|_)ie($|_)", "ie"),
    ]
    pattern, label = rules[0]
    expr = pl.when(ua.str.contains(pattern)).then(pl.lit(label))
    for pattern, label in rules[1:]:
        expr = expr.when(ua.str.contains(pattern)).then(pl.lit(label))
    return expr.otherwise(pl.lit("other"))


def _int_str(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Int64, strict=False).cast(pl.String).fill_null("na")


def clean_and_featurize(lf: pl.LazyFrame) -> pl.LazyFrame:
    ts = pl.col("timestamp").str.strip_chars()
    ua = pl.col("useragent").fill_null("").str.to_lowercase()
    day = ts.str.slice(0, 8)
    return (
        lf.with_columns(
            [pl.col(c).str.strip_chars().cast(pl.Float64, strict=False) for c in _NUMERIC_RAW]
            + [pl.col(c).str.strip_chars().fill_null("na") for c in _STRING_RAW]
            + [
                day.alias("day"),
                ts.str.slice(8, 2).cast(pl.Int32, strict=False).alias("hour"),
                day.str.strptime(pl.Date, "%Y%m%d", strict=False).dt.weekday().alias("weekday"),
                _ua_os(ua).alias("ua_os"),
                _ua_browser(ua).alias("ua_browser"),
            ]
        )
        .filter(
            pl.col("bidprice").is_not_null()
            & pl.col("payprice").is_not_null()
            & (pl.col("bidprice") > 0)
            & (pl.col("payprice") >= 0)
            & pl.col("hour").is_not_null()
            & pl.col("weekday").is_not_null()
        )
        .with_columns(
            pl.concat_str([_int_str("slotwidth"), pl.lit("x"), _int_str("slotheight")]).alias(
                "slot_size"
            ),
            (pl.col("slotwidth") * pl.col("slotheight")).alias("slot_area"),
            pl.col("adexchange").alias("report_group"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("adexchange"),
                    pl.col("slot_size"),
                    (pl.col("hour") // 6).cast(pl.String),
                ],
                separator="|",
            ).alias(_SEGMENT)
        )
    )


def drop_unwinnable(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """Keep rows where the logged value beats the market price (z < v)."""
    bad = df.filter(pl.col("payprice") >= pl.col("bidprice")).height
    if bad:
        log.warning(
            "%s: dropping %d rows (%.2f%%) with payprice >= bidprice",
            label,
            bad,
            100 * bad / max(df.height, 1),
        )
    return df.filter(pl.col("payprice") < pl.col("bidprice"))


def split_by_day(
    df: pl.DataFrame, valid_days: int, test_days: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    days = sorted(df.get_column("day").unique().drop_nulls().to_list())
    n_hold = valid_days + test_days
    if len(days) <= n_hold:
        raise ValueError(f"Only {len(days)} days {days}, need more than {n_hold} for the split")
    n = len(days)
    train_days, valid_list, test_list = (
        days[: n - n_hold],
        days[n - n_hold : n - test_days],
        days[n - test_days :],
    )
    log.info("days: train %s, valid %s, test %s", train_days, valid_list, test_list)
    return (
        df.filter(pl.col("day").is_in(train_days)),
        df.filter(pl.col("day").is_in(valid_list)),
        df.filter(pl.col("day").is_in(test_list)),
    )


def fit_vocabularies(train: pl.DataFrame, cfg: DataConfig) -> dict[str, list[str]]:
    """Category -> code mapping fitted on train only; rare and unseen values map to 0."""
    vocabs: dict[str, list[str]] = {}
    for col in [*CATEGORICAL_FEATURES, _SEGMENT]:
        counts = train.get_column(col).value_counts(sort=True)
        count_col = counts.columns[1]
        keep = counts.filter(pl.col(count_col) >= cfg.min_category_count)
        if col == "domain":
            keep = keep.head(cfg.max_domains)
        vocabs[col] = keep.get_column(col).cast(pl.String).to_list()
    return vocabs


def _encode(col: str, vocab: list[str]) -> pl.Expr:
    if not vocab:
        return pl.lit(0, dtype=pl.Int32).alias(col)
    mapping = {value: i + 1 for i, value in enumerate(vocab)}
    return pl.col(col).replace_strict(mapping, default=0, return_dtype=pl.Int32).alias(col)


def to_model_frame(df: pl.DataFrame, vocabs: dict[str, list[str]]) -> pl.DataFrame:
    return df.select(
        *[_encode(c, vocabs[c]) for c in CATEGORICAL_FEATURES],
        *[pl.col(c).cast(pl.Float32) for c in NUMERIC_FEATURES],
        _encode(_SEGMENT, vocabs[_SEGMENT]).alias("segment"),
        pl.col("report_group"),
        pl.col("bidprice").alias("v"),
        pl.col("payprice").alias("z"),
        pl.col("day"),
    )


# ----------------------------------------------------------------------------- entry points


def prepare(cfg: DataConfig) -> dict[str, Path]:
    if cfg.source == "make_ipinyou":
        train_lf, test_lf = load_make_ipinyou(cfg)
        train_all = drop_unwinnable(clean_and_featurize(train_lf).collect(), "train file")
        test = drop_unwinnable(clean_and_featurize(test_lf).collect(), "test file")
        train, valid, _ = split_by_day(train_all, cfg.valid_days, 0)
    else:
        df = drop_unwinnable(clean_and_featurize(load_raw(cfg)).collect(), "raw logs")
        train, valid, test = split_by_day(df, cfg.valid_days, cfg.test_days)

    if cfg.max_train_rows and train.height > cfg.max_train_rows:
        train = train.sample(n=cfg.max_train_rows, seed=cfg.seed)
    if cfg.max_eval_rows:
        if valid.height > cfg.max_eval_rows:
            valid = valid.sample(n=cfg.max_eval_rows, seed=cfg.seed + 1)
        if test.height > cfg.max_eval_rows:
            test = test.sample(n=cfg.max_eval_rows, seed=cfg.seed + 2)

    vocabs = fit_vocabularies(train, cfg)
    out = Path(cfg.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths, rows = {}, {}
    for name, frame in (("train", train), ("valid", valid), ("test", test)):
        model_frame = to_model_frame(frame, vocabs)
        paths[name] = out / f"{name}.parquet"
        model_frame.write_parquet(paths[name])
        rows[name] = model_frame.height
        log.info("%s: %d rows -> %s", name, model_frame.height, paths[name])

    meta = {
        "source": cfg.source,
        "advertisers": cfg.advertisers,
        "feature_names": FEATURE_NAMES,
        "categorical_features": CATEGORICAL_FEATURES,
        "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
        "rows": rows,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "vocabularies.json").write_text(json.dumps(vocabs, ensure_ascii=False), encoding="utf-8")
    return paths


def load_split(processed_dir: str | Path, name: str) -> AuctionData:
    processed_dir = Path(processed_dir)
    meta_path = processed_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found: run `bidshade prepare` first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pl.read_parquet(processed_dir / f"{name}.parquet")
    features = meta["feature_names"]
    return AuctionData(
        X=df.select(features).to_numpy().astype(np.float32),
        v=df.get_column("v").to_numpy().astype(np.float64),
        z=df.get_column("z").to_numpy().astype(np.float64),
        segment=df.get_column("segment").to_numpy().astype(np.int64),
        group=df.get_column("report_group").to_numpy().astype(str),
        feature_names=list(features),
        categorical_features=list(meta["categorical_features"]),
    )
