import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data


def test_prices_to_log_returns_matches_manual_calc():
    idx = pd.bdate_range("2021-01-01", periods=5)
    prices = pd.DataFrame({"equity": [100, 101, 99, 103, 105], "gold": [50, 50.5, 51, 50.8, 51.2]}, index=idx)
    rets = data.prices_to_log_returns(prices, ["equity", "gold"])
    expected_equity = np.log(prices["equity"]).diff()
    pd.testing.assert_series_equal(rets["equity"], expected_equity, check_names=False)
    assert rets.iloc[0].isna().all()  # first row has no prior price to diff against


def test_prices_to_simple_returns_matches_pct_change():
    idx = pd.bdate_range("2021-01-01", periods=5)
    prices = pd.DataFrame({"equity": [100, 101, 99, 103, 105]}, index=idx)
    rets = data.prices_to_simple_returns(prices, ["equity"])
    expected = prices["equity"].pct_change()
    pd.testing.assert_series_equal(rets["equity"], expected, check_names=False)


def test_simple_returns_never_produce_negative_equity_curve():
    # A genuine, permanent crash: equity curve should stay positive even
    # through a large simple-return drawdown.
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"equity": [100, 100, 60, 60, 60]}, index=idx)
    rets = data.prices_to_simple_returns(prices, ["equity"])
    equity_curve = (1 + rets["equity"].fillna(0)).cumprod()
    assert (equity_curve > 0).all()


def test_clean_price_series_repairs_spike_and_revert():
    # Synthetic reproduction of the exact bug found in production: a
    # single-day bad tick (price drops to ~1% of its neighbors) followed
    # immediately by a snap-back to the original trend.
    idx = pd.bdate_range("2019-12-16", periods=7)
    prices = pd.Series([100.0, 100.5, 101.0, 1.0, 101.3, 101.6, 101.9], index=idx)
    cleaned = data.clean_price_series(prices, ticker_label="test_ticker")
    assert cleaned.iloc[3] > 50  # bad tick repaired, no longer a near-zero print
    for i in [0, 1, 2, 4, 5, 6]:
        assert cleaned.iloc[i] == prices.iloc[i]  # everything else untouched


def test_clean_price_series_reproduces_production_glitch_shape():
    # Realistic reproduction of the exact bug found in production data: a
    # -156.8% / +156.8% log-return pair corresponds, at the price level,
    # to a ~79.2% one-day drop followed by a ~379.7% rebound back to trend.
    idx = pd.bdate_range("2019-12-16", periods=6)
    P = 100.0
    prices = pd.Series([P, P * 1.001, P * 1.002, P * (1 - 0.7918), P * 0.999, P * 1.001], index=idx)
    cleaned = data.clean_price_series(prices, ticker_label="gold_like")
    assert cleaned.iloc[3] > 90  # bad tick repaired back near trend
    for i in [0, 1, 2, 4, 5]:
        assert cleaned.iloc[i] == prices.iloc[i]  # rebound day and everything else untouched


def test_clean_price_series_leaves_genuine_crash_alone():
    # A real, non-reverting decline must not be touched by the cleaner.
    idx = pd.bdate_range("2020-03-01", periods=6)
    prices = pd.Series([100.0, 90.0, 78.0, 70.0, 68.0, 69.0], index=idx)
    cleaned = data.clean_price_series(prices, ticker_label="test_ticker")
    pd.testing.assert_series_equal(cleaned, prices)
