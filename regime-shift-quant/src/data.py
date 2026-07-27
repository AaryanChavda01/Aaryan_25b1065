"""Data acquisition and alignment.

Pulls daily prices for the configured asset classes plus a volatility
index (India VIX by default) via yfinance, caches them to disk, and
returns a single, forward-filled, date-aligned DataFrame.

Nothing in this module computes anything used for modeling — it only
fetches and aligns raw series. Feature engineering lives in features.py
so that the "what counts as training data" boundary stays obvious.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_price_series(
    series: pd.Series,
    ticker_label: str = "",
    spike_threshold: float = 0.5,
    revert_threshold: float = 0.3,
    max_run: int = 10,
) -> pd.Series:
    """Repair runs of bad ticks (stale/erroneous prints), one or more days long.

    Detects a day whose simple return exceeds ``spike_threshold`` in
    magnitude, then scans forward (up to ``max_run`` days) for the first
    later day whose price is back within ``revert_threshold`` of the
    pre-spike level — the signature of one or more bad prints that cancel
    out, rather than a genuine market move. Every day in that bad run
    (which may be a single day, or several consecutive days such as a
    decimal-shift error that persists for two or three prints before the
    feed corrects itself) is replaced by linear interpolation between the
    last good price and the recovery price.

    Genuine crashes (e.g. a real decline that does not revert within
    ``max_run`` days) are left untouched.
    """
    orig = series.copy()  # immutable reference used for detection only
    s = series.copy()     # this is what gets repaired and returned
    simple_ret = orig.pct_change()

    n = len(orig)
    i = 1
    while i < n - 1:
        r = simple_ret.iloc[i]
        if pd.isna(r) or abs(r) < spike_threshold:
            i += 1
            continue
        # Always compare against the ORIGINAL prices, never against values
        # already repaired earlier in this loop — otherwise a repaired
        # point can itself look like the start of a new "spike".
        pre_spike = orig.iloc[i - 1]
        if pre_spike == 0 or pd.isna(pre_spike):
            i += 1
            continue

        # Scan forward for the day the price recovers to near pre_spike.
        # This may be i+1 (single bad day, the original behaviour) or
        # several days later (a multi-day bad run, e.g. a decimal-shift
        # error that persists for a couple of prints).
        j = i + 1
        recovery_j = None
        while j < n and (j - i) <= max_run:
            candidate = orig.iloc[j]
            if pd.notna(candidate):
                round_trip = abs(candidate - pre_spike) / abs(pre_spike)
                if round_trip <= revert_threshold:
                    recovery_j = j
                    break
            j += 1

        if recovery_j is None:
            # No recovery within the window — treat as a genuine move.
            i += 1
            continue

        post_spike = orig.iloc[recovery_j]
        span = recovery_j - (i - 1)
        run_len = recovery_j - i
        for k in range(i, recovery_j):
            frac = (k - (i - 1)) / span
            repaired = pre_spike + frac * (post_spike - pre_spike)
            original_value = orig.iloc[k]
            s.iloc[k] = repaired
            logger.warning(
                "clean_price_series: repaired suspected bad tick for %s on %s "
                "(%.4f -> %.4f; part of a %d-day bad run)",
                ticker_label or s.name, orig.index[k], original_value, repaired, run_len,
            )

        # Resume scanning right after the repaired run so a repaired point
        # is never re-evaluated as a fresh spike.
        i = recovery_j

    return s


def prices_to_simple_returns(prices: pd.DataFrame, asset_cols) -> pd.DataFrame:
    """Simple (arithmetic) returns for the given asset columns.

    Portfolio math — weighted sums of returns and (1+r).cumprod() equity
    curves — is only valid with simple returns, not log returns. Row 0
    is NaN by construction.
    """
    rets = pd.DataFrame(index=prices.index)
    for col in asset_cols:
        rets[col] = prices[col].pct_change()
    return rets


def _cache_path(cache_dir: str, ticker: str) -> Path:
    safe = ticker.replace("^", "idx_").replace("/", "_")
    return Path(cache_dir) / f"{safe}.csv"


def _download_one(ticker: str, start: str, end: Optional[str], cache_dir: str,
                   force_refresh: bool = False) -> pd.Series:
    """Download (or load cached) adjusted close series for a single ticker."""
    path = _cache_path(cache_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_refresh:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        series = cached.iloc[:, 0]
        series.name = ticker
        logger.info("Loaded %s from cache (%d rows).", ticker, len(series))
        return series

    import yfinance as yf  # imported lazily so the rest of the package works without it

    logger.info("Downloading %s from yfinance...", ticker)
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for '{ticker}'. Check the ticker symbol "
            "and your network connection."
        )
    close_col = "Close" if "Close" in raw.columns else raw.columns[0]
    series = raw[close_col]
    if isinstance(series, pd.DataFrame):  # yfinance sometimes returns a 1-col frame
        series = series.iloc[:, 0]
    series.name = ticker
    series.to_csv(path)
    return series


def load_price_panel(
    tickers: Dict[str, str],
    vix_ticker: Optional[str],
    start: str,
    end: Optional[str] = None,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download every configured asset (and the VIX proxy) and align them.

    Returns a DataFrame indexed by trading date with one column per asset
    class name (e.g. 'equity', 'bond', 'gold') plus 'vix' if requested.
    Rows with any missing asset price are dropped (keeps the panel dense,
    which matters for covariance estimation downstream); VIX gaps are
    forward-filled since it is a side signal, not a traded asset.
    """
    series_list = []
    for asset_name, ticker in tickers.items():
        s = _download_one(ticker, start, end, cache_dir, force_refresh)
        s = clean_price_series(s, ticker_label=ticker)
        s.name = asset_name
        series_list.append(s)

    panel = pd.concat(series_list, axis=1)
    panel = panel.sort_index()
    panel = panel.dropna(how="any")

    if vix_ticker:
        vix = _download_one(vix_ticker, start, end, cache_dir, force_refresh)
        vix = clean_price_series(vix, ticker_label=vix_ticker)
        vix.name = "vix"
        panel = panel.join(vix, how="left")
        panel["vix"] = panel["vix"].ffill()

    panel.index.name = "date"
    return panel


def prices_to_log_returns(prices: pd.DataFrame, asset_cols) -> pd.DataFrame:
    """Log returns for the given asset columns. Row 0 is NaN by construction."""
    import numpy as np

    rets = pd.DataFrame(index=prices.index)
    for col in asset_cols:
        rets[col] = np.log(prices[col]).diff()
    return rets
