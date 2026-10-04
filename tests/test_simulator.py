import numpy as np

from bidshade.simulator import CPM_FEN_TO_CNY, first_price_auction, summarize


def test_win_requires_strictly_higher_bid_and_pays_own_bid():
    v = np.array([100.0, 100.0, 100.0])
    z = np.array([50.0, 50.0, 50.0])
    bids = np.array([60.0, 50.0, 40.0])
    out = first_price_auction(v, z, bids)
    assert out.won.tolist() == [True, False, False]
    assert out.paid.tolist() == [60.0, 0.0, 0.0]
    assert out.surplus.tolist() == [40.0, 0.0, 0.0]


def test_truthful_bidding_has_zero_surplus():
    v = np.array([300.0, 238.0])
    z = np.array([10.0, 200.0])
    row = summarize("truthful", v, z, v.copy())
    assert row["win_rate"] == 1.0
    assert row["surplus_cny"] == 0.0


def test_units_and_oracle_share():
    v = np.array([300.0, 300.0])
    z = np.array([100.0, 250.0])
    bids = np.array([200.0, 0.0])
    row = summarize("x", v, z, bids, oracle_surplus=400.0)
    assert np.isclose(row["spend_cny"], 200.0 * CPM_FEN_TO_CNY)
    assert np.isclose(row["surplus_cny"], 100.0 * CPM_FEN_TO_CNY)
    assert np.isclose(row["surplus_per_1k_req_fen"], 100.0 / 2)
    assert np.isclose(row["oracle_share"], 0.25)
    assert np.isclose(row["mean_bid_to_value"], 200.0 / 300.0)
