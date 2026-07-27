"""Walk-forward validation and the regime-driven backtest loop.

This is where lookahead bias either gets avoided or quietly creeps in, so
the module is structured around one rule, enforced mechanically rather
than just by convention:

    Any parameter used to make a decision for day t (scaler mean/std, HMM
    transition/emission params, state->label mapping, mu/covariance
    estimates) must be fit on data available strictly BEFORE day t.

Concretely, this means:
  * The StandardScaler, the HMM, and the volatility-based state labeling
    are all fit ONCE PER FOLD on that fold's training window only.
  * Portfolio weights decided "for day t" are computed from a trailing
    window ending at day t-1, and applied to day t's realized return.
  * Walk-forward folds are also non-overlapping in their test windows, so
    every day in the final out-of-sample curve was, at the time its
    weights were chosen, still in the model's future relative to training.

Simplification documented up front: between rebalances, weights are held
fixed rather than allowed to drift with relative asset performance
(a common simplification in pedagogical backtests — the alternative,
tracking daily drift explicitly, is a straightforward extension left as
an exercise in the README).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import optimization as opt
from . import regime as regime_mod


@dataclass
class WalkForwardConfig:
    initial_train_days: int = 756
    test_days: int = 63
    min_holding_days: int = 5
    lookback_days: int = 126


@dataclass
class BacktestResult:
    returns: pd.Series
    weights_history: pd.DataFrame
    regimes: pd.Series
    fold_transition_matrices: Dict[int, np.ndarray] = field(default_factory=dict)
    fold_state_labels: Dict[int, dict] = field(default_factory=dict)


def make_folds(n_rows: int, initial_train_days: int, test_days: int) -> List[tuple]:
    """Non-overlapping (train_start, train_end, test_start, test_end) index folds.

    train_end == test_start (exclusive/inclusive boundary), so each fold's
    training window is everything strictly before its test window. The
    training window EXPANDS fold over fold (grows to include prior test
    folds too), which is standard walk-forward practice and only adds more
    past data to the training set — never future data.
    """
    folds = []
    train_end = initial_train_days
    while train_end + test_days <= n_rows:
        test_start = train_end
        test_end = train_end + test_days
        folds.append((0, train_end, test_start, test_end))
        train_end = test_end
    # Absorb any small remainder into the final fold rather than dropping it.
    if folds and folds[-1][3] < n_rows:
        last = folds[-1]
        folds[-1] = (last[0], last[1], last[2], n_rows)
    return folds


def _fit_scaler(train_X: np.ndarray):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(train_X)
    return scaler


def run_walk_forward_backtest(
    feat_df: pd.DataFrame,
    asset_returns: pd.DataFrame,
    feature_columns: List[str],
    wf_cfg: WalkForwardConfig,
    regime_cfg: dict,
    opt_cfg: opt.OptimizationConfig,
    objectives: Dict[str, str],
    tx_cost_bps: float,
) -> BacktestResult:
    """Run the full leakage-checked walk-forward backtest.

    Parameters
    ----------
    feat_df : DataFrame
        Output of features.build_full_feature_matrix, already aligned to
        asset_returns.index (both should be dropna()'d to the same rows
        before calling this).
    asset_returns : DataFrame
        Daily simple/log returns per asset class (columns = asset names),
        same index as feat_df.
    """
    idx = feat_df.index
    n = len(idx)
    asset_names = list(asset_returns.columns)
    n_assets = len(asset_names)

    folds = make_folds(n, wf_cfg.initial_train_days, wf_cfg.test_days)
    if not folds:
        raise ValueError(
            "No walk-forward folds fit in this sample — reduce initial_train_days "
            "or test_days, or pull a longer history."
        )

    daily_returns = pd.Series(index=idx, dtype=float)
    regimes = pd.Series(index=idx, dtype=object)
    weight_rows = []
    weight_dates = []

    fold_transition_matrices: Dict[int, np.ndarray] = {}
    fold_state_labels: Dict[int, dict] = {}

    current_weights: Optional[np.ndarray] = None
    current_regime: Optional[str] = None
    days_since_rebalance = 10 ** 9  # force a rebalance on the very first test day

    for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        train_feat = feat_df.iloc[tr_s:tr_e]
        test_feat = feat_df.iloc[te_s:te_e]

        train_X_raw = train_feat[feature_columns].values
        test_X_raw = test_feat[feature_columns].values

        scaler = _fit_scaler(train_X_raw)
        train_X = scaler.transform(train_X_raw)
        test_X = scaler.transform(test_X_raw)

        rh = regime_mod.fit_and_label(
            feature_matrix=train_X,
            raw_feat_df=train_feat,
            feature_columns=feature_columns,
            n_states=regime_cfg.get("n_states", 3),
            covariance_type=regime_cfg.get("covariance_type", "diag"),
            n_iter=regime_cfg.get("n_iter", 200),
            random_state=regime_cfg.get("random_state", 42),
            vol_feature_for_labeling=regime_cfg.get("vol_feature_for_labeling", "vol_21d"),
        )
        fold_transition_matrices[fold_i] = rh.transition_matrix.copy()
        fold_state_labels[fold_i] = dict(rh.state_to_label)

        test_states = rh.predict_states(test_X)
        test_labels = rh.label(test_states)

        # Combined, chronologically-ordered index of (train tail + test) so that
        # "the day before the first test day" resolves correctly at fold seams.
        full_returns_so_far = asset_returns.iloc[: te_e]

        for offset, day_i in enumerate(range(te_s, te_e)):
            decision_i = day_i - 1  # decide using info through yesterday's close
            today_regime = test_labels[offset]
            regimes.iloc[day_i] = today_regime

            do_rebalance = (
                current_weights is None
                or (today_regime != current_regime and days_since_rebalance >= wf_cfg.min_holding_days)
            )

            if do_rebalance:
                lb_start = max(0, decision_i - wf_cfg.lookback_days + 1)
                trailing = asset_returns.iloc[lb_start: decision_i + 1]
                if len(trailing) < 5:
                    # Not enough history yet (only possible right at the very start) —
                    # hold equal weight rather than fit a degenerate covariance.
                    target_weights = np.full(n_assets, 1.0 / n_assets)
                else:
                    mu = trailing.mean().values
                    cov = trailing.cov().values
                    target_weights = opt.optimize_for_regime(
                        today_regime, mu, cov, objectives, opt_cfg
                    )

                if current_weights is None:
                    trade_cost = 0.0  # no prior position to unwind on day 1
                else:
                    turnover_amt = np.abs(target_weights - current_weights).sum()
                    trade_cost = turnover_amt * (tx_cost_bps / 10_000.0)

                current_weights = target_weights
                current_regime = today_regime
                days_since_rebalance = 0
            else:
                trade_cost = 0.0
                days_since_rebalance += 1

            day_return = float(np.dot(current_weights, asset_returns.iloc[day_i].values)) - trade_cost
            daily_returns.iloc[day_i] = day_return
            weight_rows.append(current_weights.copy())
            weight_dates.append(idx[day_i])

    weights_history = pd.DataFrame(weight_rows, index=weight_dates, columns=asset_names)

    return BacktestResult(
        returns=daily_returns.dropna(),
        weights_history=weights_history,
        regimes=regimes.dropna(),
        fold_transition_matrices=fold_transition_matrices,
        fold_state_labels=fold_state_labels,
    )


def static_benchmark_returns(asset_returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Fixed-weight, buy-and-hold-style benchmark (no periodic rebalancing cost modeled)."""
    w = np.array([weights[c] for c in asset_returns.columns])
    return asset_returns.dot(w)
