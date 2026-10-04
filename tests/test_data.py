import numpy as np
import pytest

pytest.importorskip("polars")

from bidshade.config import DataConfig  # noqa: E402
from bidshade.data import load_split, prepare  # noqa: E402
from bidshade.dataset import FEATURE_NAMES  # noqa: E402
from bidshade.synthetic import write_make_ipinyou_format  # noqa: E402


def test_prepare_on_synthetic_logs(tmp_path):
    write_make_ipinyou_format(
        tmp_path / "raw", advertiser="9999", n_train=6000, n_test=2000, seed=0
    )
    cfg = DataConfig(
        source="make_ipinyou",
        raw_dir=str(tmp_path / "raw"),
        advertisers=["9999"],
        processed_dir=str(tmp_path / "processed"),
        valid_days=1,
        min_category_count=5,
    )
    prepare(cfg)
    train, valid, test = (load_split(cfg.processed_dir, s) for s in ("train", "valid", "test"))
    assert len(train) > len(valid) > 0 and len(test) > 0
    assert train.X.shape[1] == len(FEATURE_NAMES)
    assert np.all(train.z < train.v)
    # categorical codes are non-negative ints; at least some segments are known
    cat_idx = [FEATURE_NAMES.index(c) for c in train.categorical_features]
    assert np.all(train.X[:, cat_idx] >= 0)
    assert (train.segment > 0).mean() > 0.5
    assert set(np.unique(train.group)) <= {"1", "2", "3", "4"}
    # user agent parsing: synthetic "windows_ie" etc. must not all collapse to "other"
    assert len(np.unique(train.X[:, FEATURE_NAMES.index("ua_browser")])) >= 4
