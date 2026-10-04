import numpy as np

from bidshade.landscape.base import Landscape
from bidshade.optimizer import argmax_surplus, make_q_grid, optimal_bids
from bidshade.synthetic import make_auction_data


class UniformLandscape(Landscape):
    """z ~ U(0, scale): W(b) = b / scale."""

    name = "uniform"

    def __init__(self, scale):
        self.scale = scale

    def fit(self, train, valid=None):
        return self

    def _win_prob_2d(self, X, segment, bids):
        return np.clip(bids / self.scale, 0.0, 1.0)


def test_argmax_matches_closed_form_for_uniform_prices():
    # U(b) = (v - b) * b / v is maximized at b* = v / 2
    v = np.array([100.0, 300.0])
    q = make_q_grid(0.0, 1001)
    bids = v[:, None] * q[None, :]
    best, expected = argmax_surplus(v, bids, bids / v[:, None])
    np.testing.assert_allclose(best, v / 2, atol=0.5)
    np.testing.assert_allclose(expected, v / 4, rtol=1e-3)


def test_no_bid_when_winning_is_impossible():
    v = np.array([100.0])
    bids = np.array([[10.0, 50.0, 90.0]])
    best, expected = argmax_surplus(v, bids, np.zeros_like(bids))
    assert best[0] == 0.0 and expected[0] == 0.0


def test_optimal_bids_chunking_is_consistent():
    data = make_auction_data(3000, seed=1)
    model = UniformLandscape(scale=300.0)
    q = make_q_grid(0.05, 50)
    a, _ = optimal_bids(model, data, q, chunk_size=10_000)
    b, _ = optimal_bids(model, data, q, chunk_size=257)
    np.testing.assert_array_equal(a, b)
    assert np.all(a <= data.v)
