import numpy as np
import pytest

pytest.importorskip("lightgbm")

from bidshade.censoring import CensoredSample, make_censored_sample  # noqa: E402
from bidshade.landscape import GlobalKaplanMeier  # noqa: E402
from bidshade.landscape.hazard import LightGBMDiscreteHazard  # noqa: E402
from bidshade.landscape.win_classifier import LightGBMWinClassifier  # noqa: E402
from bidshade.metrics import log_loss  # noqa: E402
from bidshade.synthetic import make_auction_data  # noqa: E402

FAST = {"learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 50, "seed": 0}


@pytest.fixture(scope="module")
def samples():
    train = make_censored_sample(make_auction_data(40_000, seed=0), 0.05, 1.0, seed=0)
    valid = make_censored_sample(
        make_auction_data(10_000, seed=1, day_offsets=(7, 8)), 0.05, 1.0, seed=1
    )
    test = make_censored_sample(
        make_auction_data(10_000, seed=2, day_offsets=(8, 10)), 0.05, 1.0, seed=2
    )
    return train, valid, test


def test_hazard_expansion_labels():
    model = LightGBMDiscreteHazard(params=FAST, bin_width=5.0)
    model.setup_bins(100.0)
    sample = CensoredSample(
        X=np.zeros((2, 1), dtype=np.float32),
        segment=np.zeros(2, dtype=np.int64),
        v=np.array([100.0, 100.0]),
        bid=np.array([30.0, 12.0]),
        won=np.array([True, False]),
        price_obs=np.array([12.0, np.nan]),
        feature_names=["f"],
        categorical_features=[],
    )
    design, labels = model.expand(sample)
    # win with z=12: bins 0,1,2 (event in bin 2); loss at 12: bins 0,1 fully below the bid
    assert labels.tolist() == [False, False, True, False, False]
    assert design[:, -1].tolist() == [0.0, 5.0, 10.0, 0.0, 5.0]


def test_win_classifier_is_monotone_and_beats_global_km(samples):
    train, valid, test = samples
    model = LightGBMWinClassifier(FAST, num_boost_round=200, early_stopping_rounds=20).fit(
        train, valid
    )
    grid = np.tile(np.linspace(0, 300, 31), (200, 1))
    w = model.win_prob(test.X[:200], test.segment[:200], grid)
    assert np.all(np.diff(w, axis=1) >= -1e-9)
    km = GlobalKaplanMeier().fit(train)
    p_model = model.win_prob(test.X, test.segment, test.bid)
    p_km = km.win_prob(test.X, test.segment, test.bid)
    assert log_loss(test.won, p_model) < log_loss(test.won, p_km)


def test_hazard_gives_valid_cdf(samples):
    train, valid, test = samples
    model = LightGBMDiscreteHazard(
        FAST,
        bin_width=5.0,
        num_boost_round=150,
        early_stopping_rounds=20,
        max_train_requests=20_000,
    )
    model.fit(train, valid)
    grid = np.tile(np.linspace(0, 400, 41), (100, 1))
    w = model.win_prob(test.X[:100], test.segment[:100], grid)
    assert np.all(w >= 0) and np.all(w <= 1)
    assert np.all(np.diff(w, axis=1) >= -1e-9)
    assert np.allclose(w[:, 0], 0.0)
