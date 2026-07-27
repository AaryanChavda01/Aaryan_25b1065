import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features


def _sample_prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    log_rets = rng.normal(0.0003, 0.012, n)
    prices = 100 * np.exp(np.cumsum(log_rets))
    return pd.Series(prices, index=idx, name="equity")


def test_features_are_causal_truncation_invariant():
    """The value of a rolling feature at time t must depend only on data up
    to and including t — truncating the series after t must not change it.
    This is the core anti-lookahead property every feature must satisfy.
    """
    prices = _sample_prices(300)
    full = features.build_asset_features(prices, momentum_windows=(5, 21), volatility_windows=(21,))

    cutoff = 200
    truncated_prices = prices.iloc[: cutoff + 1]
    truncated = features.build_asset_features(
        truncated_prices, momentum_windows=(5, 21), volatility_windows=(21,)
    )

    row_full = full.iloc[cutoff]
    row_trunc = truncated.iloc[-1]

    for col in full.columns:
        a, b = row_full[col], row_trunc[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, f"Feature '{col}' leaked future information at row {cutoff}"


def test_momentum_window_matches_manual_log_diff():
    prices = _sample_prices(100)
    feat = features.build_asset_features(prices, momentum_windows=(21,), volatility_windows=(21,))
    manual = np.log(prices).diff(21)
    pd.testing.assert_series_equal(feat["mom_21d"], manual, check_names=False)


def test_volatility_feature_is_rolling_std_of_log_returns():
    prices = _sample_prices(150)
    feat = features.build_asset_features(prices, momentum_windows=(5,), volatility_windows=(21,))
    manual_vol = np.log(prices).diff().rolling(21).std()
    pd.testing.assert_series_equal(feat["vol_21d"], manual_vol, check_names=False)


def test_sanity_check_spikes_returns_expected_columns():
    prices = _sample_prices(300)
    feat = features.build_asset_features(prices, momentum_windows=(21,), volatility_windows=(21,))
    stress = [(prices.index[50], prices.index[70])]
    out = features.sanity_check_spikes(feat, "vol_21d", stress)
    assert "avg_vol_21d" in out.columns
    assert len(out) == 1
