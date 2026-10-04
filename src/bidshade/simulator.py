"""First-price auction replay against logged competing prices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Prices are fen (0.01 CNY) per 1000 impressions: one impression at CPM p costs p / 1000 fen.
CPM_FEN_TO_CNY = 1e-3 * 1e-2


@dataclass
class AuctionOutcome:
    won: np.ndarray
    paid: np.ndarray  # CPM price paid per request (0 for losses)
    surplus: np.ndarray  # v - b for wins, 0 for losses (CPM units)


def first_price_auction(v: np.ndarray, z: np.ndarray, bids: np.ndarray) -> AuctionOutcome:
    """We win iff our bid beats the best competitor strictly, and pay our own bid."""
    v, z, bids = (np.asarray(a, dtype=np.float64) for a in (v, z, bids))
    won = bids > z
    return AuctionOutcome(
        won=won,
        paid=np.where(won, bids, 0.0),
        surplus=np.where(won, v - bids, 0.0),
    )


def summarize(
    name: str,
    v: np.ndarray,
    z: np.ndarray,
    bids: np.ndarray,
    oracle_surplus: float | None = None,
) -> dict:
    out = first_price_auction(v, z, bids)
    n = len(v)
    wins = int(out.won.sum())
    surplus_total = float(out.surplus.sum())
    active = bids > 0
    return {
        "strategy": name,
        "requests": n,
        "wins": wins,
        "win_rate": wins / n if n else float("nan"),
        "spend_cny": float(out.paid.sum()) * CPM_FEN_TO_CNY,
        "surplus_cny": surplus_total * CPM_FEN_TO_CNY,
        "avg_paid_cpm_fen": float(out.paid[out.won].mean()) if wins else float("nan"),
        # surplus per request in fen is (v - b) / 1000, so per 1000 requests it is sum(v - b) / n
        "surplus_per_1k_req_fen": surplus_total / n if n else float("nan"),
        "mean_bid_to_value": float(np.mean(bids[active] / v[active])) if active.any() else 0.0,
        "oracle_share": surplus_total / oracle_surplus if oracle_surplus else float("nan"),
    }
