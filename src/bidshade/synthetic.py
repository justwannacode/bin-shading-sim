"""Synthetic iPinYou-like logs with a known, feature-dependent market price distribution.

Two uses:
  * `write_make_ipinyou_format` writes TSV files in make-ipinyou-data layout, so the whole
    pipeline (including polars preprocessing) can be smoke-tested without the real dataset;
  * `make_auction_data` builds AuctionData in memory (numpy only) for fast unit tests.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from .dataset import CATEGORICAL_FEATURES, FEATURE_NAMES, AuctionData
from .schema import MAKE_IPINYOU_COLUMNS

SIZES = [(300, 250), (728, 90), (160, 600), (336, 280), (950, 90), (468, 60)]
SIZE_P = [0.30, 0.25, 0.15, 0.12, 0.10, 0.08]
SIZE_EFFECT = np.array([0.35, 0.10, 0.20, 0.25, -0.20, -0.35])
EXCHANGES = ["1", "2", "3", "4"]
EXCHANGE_P = [0.40, 0.30, 0.20, 0.10]
EXCHANGE_EFFECT = np.array([0.15, -0.10, 0.05, -0.30])
VISIBILITY = ["FirstView", "SecondView", "OtherView", "Na"]
VISIBILITY_EFFECT = np.array([0.25, -0.05, -0.15, 0.0])
FORMATS = ["Na", "Fixed", "Pop"]
USER_AGENTS = [
    "windows_ie", "windows_chrome", "windows_firefox", "windows_maxthon",
    "mac_safari", "android_safari", "ios_safari", "linux_firefox",
]  # fmt: skip
VALUES = np.array([227.0, 238.0, 300.0])
N_DOMAINS = 400


def generate_logs(n: int, seed: int, day_offsets: tuple[int, int], start: str = "20130606") -> dict:
    """Raw-like columns for n rows with days in [day_offsets[0], day_offsets[1])."""
    rng = np.random.default_rng(seed)
    n_draw = int(n * 1.3) + 100  # oversample: rows with z >= v are dropped below
    day = rng.integers(day_offsets[0], day_offsets[1], n_draw)
    hour = rng.integers(0, 24, n_draw)
    exch = rng.choice(len(EXCHANGES), n_draw, p=EXCHANGE_P)
    size = rng.choice(len(SIZES), n_draw, p=SIZE_P)
    vis = rng.integers(0, len(VISIBILITY), n_draw)
    fmt = rng.integers(0, len(FORMATS), n_draw)
    ua = rng.integers(0, len(USER_AGENTS), n_draw)
    domain = (rng.zipf(1.4, n_draw) - 1) % N_DOMAINS
    region = rng.choice(np.arange(1, 400, 15), n_draw)
    city = region * 10 + rng.integers(0, 5, n_draw)
    slotprice = rng.choice([0, 0, 0, 5, 10, 30, 50, 80], n_draw).astype(np.float64)
    value = VALUES[rng.choice(3, n_draw, p=[0.2, 0.2, 0.6])]

    domain_effect = np.random.default_rng(12345).normal(0.0, 0.35, N_DOMAINS)
    mu = (
        3.9
        + EXCHANGE_EFFECT[exch]
        + SIZE_EFFECT[size]
        + VISIBILITY_EFFECT[vis]
        + 0.25 * np.sin(2 * np.pi * (hour - 14) / 24)
        + domain_effect[domain]
    )
    z = np.floor(np.exp(mu + 0.55 * rng.standard_normal(n_draw)))
    z = np.maximum(z, slotprice)
    keep = np.flatnonzero(z < value)[:n]

    start_date = dt.datetime.strptime(start, "%Y%m%d").date()
    dates = [start_date + dt.timedelta(days=int(d)) for d in day[keep]]
    minute, second, ms = (rng.integers(0, m, len(keep)) for m in (60, 60, 1000))
    timestamps = [
        f"{d:%Y%m%d}{h:02d}{mi:02d}{s:02d}{x:03d}"
        for d, h, mi, s, x in zip(dates, hour[keep], minute, second, ms, strict=True)
    ]
    return {
        "timestamp": np.array(timestamps),
        "weekday": np.array([d.isoweekday() for d in dates]),
        "hour": hour[keep],
        "useragent": np.array(USER_AGENTS)[ua[keep]],
        "region": region[keep],
        "city": city[keep],
        "adexchange": np.array(EXCHANGES)[exch[keep]],
        "domain": np.array([f"d{d:04d}" for d in domain[keep]]),
        "slotwidth": np.array([SIZES[s][0] for s in size[keep]]),
        "slotheight": np.array([SIZES[s][1] for s in size[keep]]),
        "slotvisibility": np.array(VISIBILITY)[vis[keep]],
        "slotformat": np.array(FORMATS)[fmt[keep]],
        "slotprice": slotprice[keep],
        "bidprice": value[keep],
        "payprice": z[keep],
    }


def write_make_ipinyou_format(
    root: str | Path,
    advertiser: str = "9999",
    n_train: int = 200_000,
    n_test: int = 80_000,
    seed: int = 0,
) -> Path:
    """Write <root>/<advertiser>/{train,test}.log.txt: 7 train days, then 3 test days."""
    out = Path(root) / advertiser
    out.mkdir(parents=True, exist_ok=True)
    for split, n, days, s in (
        ("train", n_train, (0, 7), seed),
        ("test", n_test, (7, 10), seed + 1),
    ):
        cols = generate_logs(n, s, days)
        m = len(cols["timestamp"])
        rng = np.random.default_rng(s + 100)
        filler = {
            "click": (rng.random(m) < 0.001).astype(int),
            "bidid": [f"b{s}_{i}" for i in range(m)],
            "logtype": np.ones(m, dtype=int),
            "ipinyouid": [f"u{i % 50000}" for i in range(m)],
            "ip": ["1.2.3.*"] * m,
            "url": ["null"] * m,
            "urlid": ["null"] * m,
            "slotid": [f"slot_{i % 300}" for i in range(m)],
            "creative": ["c1"] * m,
            "keypage": ["k1"] * m,
            "advertiser": [advertiser] * m,
            "usertag": ["null"] * m,
        }
        table = {**filler, **cols}
        columns = [[str(x) for x in table[name]] for name in MAKE_IPINYOU_COLUMNS]
        with open(out / f"{split}.log.txt", "w", encoding="utf-8") as f:
            f.write("\t".join(MAKE_IPINYOU_COLUMNS) + "\n")
            for row in zip(*columns, strict=True):
                f.write("\t".join(row) + "\n")
    return out


def _codes(values: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(values.astype(str), return_inverse=True)
    return inverse + 1  # 0 is reserved for rare / unknown


def make_auction_data(n: int, seed: int = 0, day_offsets: tuple[int, int] = (0, 7)) -> AuctionData:
    """AuctionData directly from synthetic logs (mirrors the polars feature pipeline)."""
    c = generate_logs(n, seed, day_offsets)
    os_browser = np.char.partition(c["useragent"].astype(str), "_")
    slot_size = np.char.add(
        np.char.add(c["slotwidth"].astype(str), "x"), c["slotheight"].astype(str)
    )
    categorical = {
        "region": c["region"],
        "city": c["city"],
        "adexchange": c["adexchange"],
        "domain": c["domain"],
        "slot_size": slot_size,
        "slotvisibility": c["slotvisibility"],
        "slotformat": c["slotformat"],
        "ua_os": os_browser[:, 0],
        "ua_browser": os_browser[:, 2],
        "advertiser": np.full(len(c["hour"]), "9999"),
    }
    numeric = {
        "hour": c["hour"],
        "weekday": c["weekday"],
        "slotwidth": c["slotwidth"],
        "slotheight": c["slotheight"],
        "slot_area": c["slotwidth"] * c["slotheight"],
        "slotprice": c["slotprice"],
    }
    columns = {**{k: _codes(v) for k, v in categorical.items()}, **numeric}
    X = np.column_stack([columns[name] for name in FEATURE_NAMES]).astype(np.float32)
    segment_key = np.char.add(
        np.char.add(np.char.add(c["adexchange"].astype(str), "|"), slot_size),
        np.char.add("|", (c["hour"] // 6).astype(str)),
    )
    return AuctionData(
        X=X,
        v=c["bidprice"].astype(np.float64),
        z=c["payprice"].astype(np.float64),
        segment=_codes(segment_key).astype(np.int64),
        group=c["adexchange"].astype(str),
        feature_names=list(FEATURE_NAMES),
        categorical_features=list(CATEGORICAL_FEATURES),
    )
