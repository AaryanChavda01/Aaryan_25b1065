"""Regime-conditional convex portfolio optimization.

Given a mean-return vector and covariance matrix estimated from a
TRAILING (already-realized) window of returns, solve for asset weights
appropriate to the currently-detected regime:

  - Bull    -> maximize a Sharpe-like objective (return per unit risk)
  - Bear    -> minimize volatility subject to a modest target return
  - Crisis  -> minimize volatility outright (capital preservation)

All three are solved as convex programs with cvxpy. Maximizing the true
Sharpe ratio is not itself convex, so "max_sharpe" is implemented with the
standard trick: fix a target excess return of 1 (via a scaling variable),
minimize variance, then re-normalize the resulting direction back onto the
simplex — this is equivalent to maximizing return/risk for long-only,
fully-invested portfolios under the usual regularity conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class OptimizationConfig:
    max_weight: float = 0.70
    min_weight: float = 0.0
    risk_free_rate_annual: float = 0.06
    bear_target_annual_return: float = 0.03
    trading_days: int = 252


def _bounds_constraints(w, cfg: OptimizationConfig):
    import cvxpy as cp

    return [cp.sum(w) == 1, w >= cfg.min_weight, w <= cfg.max_weight]


def _solve(problem):
    """Solve a cvxpy Problem without assuming any one solver is installed.

    Different cvxpy installs ship with different bundled solvers (ECOS is
    no longer bundled by default on every platform/Python version — this
    is exactly what broke on Python 3.14 with only a subset of solvers
    available). Try the solver-free default first (cvxpy auto-selects from
    whatever's installed), then fall back through a list of common convex
    solvers, skipping any that aren't present or that fail on this problem.
    """
    import cvxpy as cp

    try:
        problem.solve()
        if problem.status in ("optimal", "optimal_inaccurate"):
            return
    except Exception:
        pass

    for solver_name in ("CLARABEL", "ECOS", "SCS", "OSQP", "CVXOPT"):
        solver = getattr(cp, solver_name, None)
        if solver is None:
            continue
        try:
            problem.solve(solver=solver)
            if problem.status in ("optimal", "optimal_inaccurate"):
                return
        except Exception:
            continue
    # If nothing worked, problem.value/variable.value are left as None and
    # the caller's _clean_weights() fallback (equal weight) takes over.


def min_vol_weights(cov: np.ndarray, cfg: OptimizationConfig) -> np.ndarray:
    import cvxpy as cp

    n = cov.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = _bounds_constraints(w, cfg)
    problem = cp.Problem(objective, constraints)
    _solve(problem)
    return _clean_weights(w.value, n)


def min_vol_target_return_weights(mu: np.ndarray, cov: np.ndarray, target_annual_return: float,
                                   cfg: OptimizationConfig) -> np.ndarray:
    import cvxpy as cp

    n = cov.shape[0]
    target_daily = target_annual_return / cfg.trading_days
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = _bounds_constraints(w, cfg) + [mu @ w >= target_daily]
    problem = cp.Problem(objective, constraints)
    _solve(problem)
    if w.value is None:
        # Target infeasible with these bounds (e.g. every asset expects a loss) —
        # fall back to unconstrained-return min-vol so the backtest never crashes.
        return min_vol_weights(cov, cfg)
    return _clean_weights(w.value, n)


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, cfg: OptimizationConfig) -> np.ndarray:
    """Long-only, fully-invested max-Sharpe via the standard convex reformulation.

    Solve: minimize y^T Sigma y  s.t. (mu - rf)^T y == 1, y >= 0
    then w = y / sum(y). This recovers the tangency-style max-Sharpe
    portfolio without needing a non-convex ratio objective. If no asset
    has positive expected excess return the reformulation is infeasible;
    we fall back to min-vol in that case.
    """
    import cvxpy as cp

    n = cov.shape[0]
    rf_daily = cfg.risk_free_rate_annual / cfg.trading_days
    excess = mu - rf_daily

    if np.all(excess <= 0):
        return min_vol_weights(cov, cfg)

    y = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov)))
    constraints = [excess @ y == 1, y >= 0]
    problem = cp.Problem(objective, constraints)
    _solve(problem)

    if y.value is None or y.value.sum() <= 0:
        return min_vol_weights(cov, cfg)

    w = y.value / y.value.sum()
    # Re-apply the per-asset cap post-hoc via a light projection, then
    # re-solve min-vol with that cap if the raw solution violates it.
    if np.any(w > cfg.max_weight + 1e-6):
        return min_vol_target_return_weights(mu, cov, target_annual_return=_annualize_mean(mu, cfg),
                                              cfg=cfg)
    return _clean_weights(w, n)


def _annualize_mean(mu: np.ndarray, cfg: OptimizationConfig) -> float:
    # A modest achievable target: the average daily mean return across assets, annualized.
    return float(np.mean(mu) * cfg.trading_days)


def _clean_weights(w: Optional[np.ndarray], n: int) -> np.ndarray:
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.clip(w, 0, None)
    total = w.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return w / total


def optimize_for_regime(regime_label: str, mu: np.ndarray, cov: np.ndarray,
                         objectives: Dict[str, str], cfg: OptimizationConfig) -> np.ndarray:
    """Dispatch to the objective configured for this regime label."""
    objective = objectives.get(regime_label, "min_vol")
    if objective == "max_sharpe":
        return max_sharpe_weights(mu, cov, cfg)
    if objective == "min_vol_target_return":
        return min_vol_target_return_weights(mu, cov, cfg.bear_target_annual_return, cfg)
    if objective == "min_vol":
        return min_vol_weights(cov, cfg)
    raise ValueError(f"Unknown objective '{objective}' for regime '{regime_label}'.")
