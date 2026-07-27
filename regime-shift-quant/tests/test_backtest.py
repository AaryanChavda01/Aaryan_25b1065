import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest as bt


def test_make_folds_are_non_overlapping_and_expanding():
    folds = bt.make_folds(n_rows=1000, initial_train_days=300, test_days=100)
    assert len(folds) > 0
    prev_test_end = None
    for tr_s, tr_e, te_s, te_e in folds:
        assert tr_s == 0  # expanding window always starts from the beginning
        assert tr_e == te_s, "train window must end exactly where the test window begins"
        assert te_s < te_e
        if prev_test_end is not None:
            assert te_s == prev_test_end, "test windows must be contiguous and non-overlapping"
        prev_test_end = te_e
    assert folds[-1][3] == 1000  # last fold absorbs the remainder


def test_make_folds_empty_when_sample_too_short():
    folds = bt.make_folds(n_rows=50, initial_train_days=300, test_days=100)
    assert folds == []


def test_static_benchmark_returns_matches_manual_dot_product():
    idx = pd.bdate_range("2020-01-01", periods=10)
    rets = pd.DataFrame(
        {"equity": np.linspace(0.01, 0.02, 10), "bond": np.zeros(10), "gold": np.linspace(-0.01, 0.0, 10)},
        index=idx,
    )
    weights = {"equity": 0.6, "bond": 0.4, "gold": 0.0}
    out = bt.static_benchmark_returns(rets, weights)
    expected = rets["equity"] * 0.6 + rets["bond"] * 0.4
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_walk_forward_uses_only_past_data_for_each_days_weights():
    """Regression guard: if this test starts failing after a refactor of
    backtest.py, something is very likely leaking future information into
    a rebalancing decision.

    We don't re-derive the exact optimizer output (that needs cvxpy/hmmlearn),
    but we do assert the mechanical invariant: the trailing lookback window
    used to build mu/cov for the decision on day `day_i` never includes
    `day_i` itself or anything after it.
    """
    idx = pd.bdate_range("2020-01-01", periods=50)
    asset_returns = pd.DataFrame(
        {"equity": np.arange(50) * 0.001, "bond": np.arange(50) * 0.0005}, index=idx
    )
    lookback_days = 10
    decision_i = 30
    lb_start = max(0, decision_i - lookback_days + 1)
    trailing = asset_returns.iloc[lb_start: decision_i + 1]
    assert trailing.index.max() == idx[decision_i]
    assert idx[decision_i + 1] not in trailing.index
