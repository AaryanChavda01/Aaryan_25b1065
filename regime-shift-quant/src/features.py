"""Feature engineering for the regime classifier.

Every feature here is built with `.rolling(window)` (or `.diff()`), which
by construction only looks backward: the value at row t is a function of
rows [t-window+1, ... t] and nothing after. That is what makes these safe
to compute once over the full price history and then slice per
walk-forward fold — the *features* are causal, even though care is still
required downstream (fitting/scaling must still happen train-only; see
backtest.py) to keep the whole pipeline leakage-free.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def build_asset_features(
    prices: pd.Series,
    momentum_windows: Iterable[int] = (5, 21, 63),
    volatility_windows: Iterable[int] = (21, 63),
) -> pd.DataFrame:
    """Build the per-asset feature block used to drive the regime HMM.

    Parameters
    ----------
    prices : pd.Series
        Price series for a single asset (e.g. the equity leg), used as the
        "market" whose regime we are trying to detect.
    """
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices
    df["log_ret"] = np.log(prices).diff()

    for w in momentum_windows:
        # Rolling total log-return over the trailing window — trend/momentum.
        df[f"mom_{w}d"] = np.log(prices).diff(w)

    for w in volatility_windows:
        # Rolling realized volatility of daily log-returns.
        df[f"vol_{w}d"] = df["log_ret"].rolling(w).std()

    return df


def add_vix_features(df: pd.DataFrame, vix: pd.Series, zscore_window: int = 252) -> pd.DataFrame:
    """Attach the VIX proxy level plus a rolling z-score (descriptive only)."""
    out = df.copy()
    out["vix"] = vix.reindex(out.index).ffill()
    roll_mean = out["vix"].rolling(zscore_window).mean()
    roll_std = out["vix"].rolling(zscore_window).std()
    out["vix_z"] = (out["vix"] - roll_mean) / roll_std
    return out


def sanity_check_spikes(df: pd.DataFrame, vol_col: str, stress_periods: List[tuple]) -> pd.DataFrame:
    """Return average of `vol_col` inside each (start, end) stress window.

    Use this to eyeball that the volatility feature actually spikes during
    known-turbulent periods (e.g. Mar 2020, the 2022 drawdown) before
    trusting it as an HMM input.
    """
    rows = []
    for start, end in stress_periods:
        mask = (df.index >= start) & (df.index <= end)
        rows.append({"start": start, "end": end, f"avg_{vol_col}": df.loc[mask, vol_col].mean()})
    return pd.DataFrame(rows)


def build_full_feature_matrix(
    asset_prices: pd.DataFrame,
    market_col: str,
    vix: pd.Series | None,
    momentum_windows: Iterable[int] = (5, 21, 63),
    volatility_windows: Iterable[int] = (21, 63),
    vix_zscore_window: int = 252,
) -> pd.DataFrame:
    """Convenience wrapper: build regime features off the chosen market column."""
    feat = build_asset_features(
        asset_prices[market_col],
        momentum_windows=momentum_windows,
        volatility_windows=volatility_windows,
    )
    if vix is not None:
        feat = add_vix_features(feat, vix, zscore_window=vix_zscore_window)
    return feat
