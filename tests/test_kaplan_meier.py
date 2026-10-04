import numpy as np
import pytest

from bidshade.censoring import make_censored_sample
from bidshade.landscape import GlobalKaplanMeier, KMCurve, SegmentedKaplanMeier
from bidshade.synthetic import make_auction_data


def test_km_without_censoring_is_empirical_cdf():
    z = np.array([10.0, 20.0, 20.0, 30.0])
    curve = KMCurve.fit(z, np.ones(4, dtype=bool))
    # W(b) = P(z < b)
    np.testing.assert_allclose(
        curve.win_prob(np.array([5, 10, 10.5, 20, 25, 31])), [0, 0, 0.25, 0.25, 0.75, 1.0]
    )


def test_km_recovers_distribution_under_censoring():
    rng = np.random.default_rng(0)
    z = rng.exponential(50.0, 200_000)
    bid = rng.uniform(0, 300, z.size)
    won = bid > z
    curve = KMCurve.fit(np.where(won, z, bid), won)
    b = np.array([20.0, 50.0, 100.0, 150.0])
    np.testing.assert_allclose(curve.win_prob(b), 1 - np.exp(-b / 50.0), atol=0.01)


def test_km_matches_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(1)
    z = np.floor(rng.lognormal(4.0, 0.6, 5000))
    bid = rng.uniform(0, 300, z.size)
    won = bid > z
    durations = np.where(won, z, bid)
    ours = KMCurve.fit(durations, won)
    kmf = lifelines.KaplanMeierFitter().fit(durations, event_observed=won)
    times = ours.event_times
    theirs = kmf.survival_function_at_times(times).to_numpy()
    np.testing.assert_allclose(ours.survival, theirs, atol=1e-10)


def test_segmented_km_falls_back_to_global():
    data = make_auction_data(20_000, seed=2)
    cs = make_censored_sample(data, 0.05, 1.0, seed=0)
    seg = SegmentedKaplanMeier(min_size=500).fit(cs)
    glob = GlobalKaplanMeier().fit(cs)
    assert seg.n_segments > 0
    bids = np.full(3, 80.0)
    unknown = np.zeros(3, dtype=np.int64)  # code 0 = rare segment
    np.testing.assert_allclose(
        seg.win_prob(data.X[:3], unknown, bids), glob.win_prob(data.X[:3], unknown, bids)
    )
