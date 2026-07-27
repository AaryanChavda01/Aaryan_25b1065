"""Performance metrics for backtested return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    if len(equity) == 0 or equity.iloc[-1] <= 0:
        return np.nan
    n_years = len(returns) / trading_days
    if n_years <= 0:
        return np.nan
    return float(equity.iloc[-1] ** (1 / n_years) - 1)


def annualized_vol(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=0) * np.sqrt(trading_days))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0,
                  trading_days: int = TRADING_DAYS) -> float:
    rf_daily = risk_free_annual / trading_days
    excess = returns - rf_daily
    denom = excess.std(ddof=0)
    # Guard against floating-point noise on near-constant series (e.g. exact
    # ties in a toy/test return stream), not just an exact-zero denominator.
    if np.isnan(denom) or denom < 1e-12:
        return np.nan
    return float(excess.mean() / denom * np.sqrt(trading_days))


def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.0,
                   trading_days: int = TRADING_DAYS) -> float:
    rf_daily = risk_free_annual / trading_days
    excess = returns - rf_daily
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=0)
    if len(downside) == 0 or np.isnan(downside_std) or downside_std < 1e-12:
        return np.nan
    return float(excess.mean() / downside_std * np.sqrt(trading_days))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(cagr(returns, trading_days) / abs(mdd))


def turnover(weights_history: pd.DataFrame) -> float:
    """Average one-way turnover per rebalance: mean(sum(|w_t - w_{t-1}|) / 2)."""
    diffs = weights_history.diff().abs().sum(axis=1) / 2
    return float(diffs.mean())


def annualized_turnover(weights_history: pd.DataFrame, rebalances_per_year: float) -> float:
    return turnover(weights_history) * rebalances_per_year


def summarize(returns: pd.Series, weights_history: pd.DataFrame | None = None,
              risk_free_annual: float = 0.0, trading_days: int = TRADING_DAYS,
              rebalances_per_year: float | None = None) -> dict:
    """One-stop performance summary row for a strategy's return series."""
    out = {
        "CAGR": cagr(returns, trading_days),
        "Ann.Vol": annualized_vol(returns, trading_days),
        "Sharpe": sharpe_ratio(returns, risk_free_annual, trading_days),
        "Sortino": sortino_ratio(returns, risk_free_annual, trading_days),
        "MaxDrawdown": max_drawdown(returns),
        "Calmar": calmar_ratio(returns, trading_days),
    }
    if weights_history is not None and len(weights_history) > 1:
        out["AvgTurnoverPerRebalance"] = turnover(weights_history)
        if rebalances_per_year:
            out["AnnualizedTurnover"] = annualized_turnover(weights_history, rebalances_per_year)
    return out
