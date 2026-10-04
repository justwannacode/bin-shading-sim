import numpy as np

from bidshade.censoring import make_censored_sample
from bidshade.synthetic import make_auction_data


def test_censored_sample_hides_price_for_losses():
    data = make_auction_data(5000, seed=0)
    cs = make_censored_sample(data, 0.05, 1.0, replicates=2, seed=3)
    assert len(cs) == 2 * len(data)
    z = np.repeat(data.z, 2)
    assert np.array_equal(cs.won, cs.bid > z)
    assert np.all(np.isnan(cs.price_obs[~cs.won]))
    assert np.array_equal(cs.price_obs[cs.won], z[cs.won])
    # survival view: event at z for wins, censored at the bid for losses
    assert np.array_equal(cs.durations[~cs.won], cs.bid[~cs.won])
    assert 0.2 < cs.win_rate < 0.95


def test_bids_stay_within_alpha_range():
    data = make_auction_data(2000, seed=0)
    cs = make_censored_sample(data, 0.2, 0.6, seed=0)
    ratio = cs.bid / cs.v
    assert ratio.min() >= 0.2 and ratio.max() <= 0.6


def test_drop_features_removes_columns():
    data = make_auction_data(500, seed=0)
    reduced = data.drop_features(["weekday", "domain"])
    assert reduced.X.shape[1] == data.X.shape[1] - 2
    assert "weekday" not in reduced.feature_names
    assert "domain" not in reduced.categorical_features
    col = data.feature_names.index("slotprice")
    np.testing.assert_array_equal(
        reduced.X[:, reduced.feature_names.index("slotprice")], data.X[:, col]
    )
