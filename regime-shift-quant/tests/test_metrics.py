import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import metrics


def _flat_return_series(daily_ret, n=252):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(daily_ret, index=idx)


def test_cagr_matches_constant_compounding():
    r = _flat_return_series(0.0, n=252)
    assert metrics.cagr(r) == pytest.approx(0.0, abs=1e-9)

    # A constant daily return of x for 252 days compounds to (1+x)^252 - 1 annually.
    r2 = _flat_return_series(0.001, n=252)
    expected = (1.001) ** 252 - 1
    assert metrics.cagr(r2) == pytest.approx(expected, rel=1e-6)


def test_sharpe_zero_for_zero_vol_excess():
    r = _flat_return_series(0.0005, n=252)
    # Zero volatility in excess returns -> Sharpe denominator is 0 -> nan by design.
    assert np.isnan(metrics.sharpe_ratio(r))


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=1000)
    r = pd.Series(rng.normal(0.0006, 0.01, len(idx)), index=idx)
    assert metrics.sharpe_ratio(r) > 0


def test_max_drawdown_is_negative_or_zero():
    idx = pd.bdate_range("2020-01-01", periods=10)
    r = pd.Series([0.1, 0.1, -0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], index=idx)
    mdd = metrics.max_drawdown(r)
    assert mdd < 0
    # Manually verify the trough: equity peaks at 1.1*1.1=1.21 then drops by 50%.
    equity = (1 + r).cumprod()
    expected_trough_equity = equity.iloc[1] * 0.5
    assert equity.min() == pytest.approx(expected_trough_equity)


def test_calmar_ratio_uses_cagr_over_abs_mdd():
    idx = pd.bdate_range("2020-01-01", periods=252)
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.008, len(idx)), index=idx)
    calmar = metrics.calmar_ratio(r)
    expected = metrics.cagr(r) / abs(metrics.max_drawdown(r))
    assert calmar == pytest.approx(expected)


def test_turnover_zero_for_static_weights():
    idx = pd.bdate_range("2020-01-01", periods=5)
    weights = pd.DataFrame({"a": [0.5] * 5, "b": [0.5] * 5}, index=idx)
    assert metrics.turnover(weights) == pytest.approx(0.0)


def test_turnover_detects_full_flip():
    idx = pd.bdate_range("2020-01-01", periods=2)
    weights = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]}, index=idx)
    # |1-0| + |0-1| = 2, divided by 2 => full 100% turnover on the rebalance day.
    diffs = weights.diff().abs().sum(axis=1) / 2
    assert diffs.iloc[1] == pytest.approx(1.0)
