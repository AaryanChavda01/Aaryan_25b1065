import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cp = pytest.importorskip("cvxpy", reason="cvxpy not installed")

from src import optimization as opt


@pytest.fixture
def cfg():
    return opt.OptimizationConfig(max_weight=0.7, min_weight=0.0, risk_free_rate_annual=0.06)


def _toy_mu_cov():
    mu = np.array([0.0006, 0.0002, 0.0001])
    cov = np.array(
        [
            [0.00040, -0.00005, 0.00002],
            [-0.00005, 0.00010, 0.00001],
            [0.00002, 0.00001, 0.00015],
        ]
    )
    return mu, cov


def test_weights_sum_to_one_and_respect_bounds(cfg):
    mu, cov = _toy_mu_cov()
    for fn in [
        lambda: opt.min_vol_weights(cov, cfg),
        lambda: opt.max_sharpe_weights(mu, cov, cfg),
        lambda: opt.min_vol_target_return_weights(mu, cov, 0.03, cfg),
    ]:
        w = fn()
        assert w.sum() == pytest.approx(1.0, abs=1e-4)
        assert (w >= cfg.min_weight - 1e-6).all()
        assert (w <= cfg.max_weight + 1e-6).all()


def test_min_vol_is_lower_variance_than_equal_weight(cfg):
    mu, cov = _toy_mu_cov()
    w_minvol = opt.min_vol_weights(cov, cfg)
    w_eq = np.array([1 / 3, 1 / 3, 1 / 3])
    var_minvol = w_minvol @ cov @ w_minvol
    var_eq = w_eq @ cov @ w_eq
    assert var_minvol <= var_eq + 1e-9


def test_optimize_for_regime_dispatches_correctly(cfg):
    mu, cov = _toy_mu_cov()
    objectives = {"Bull": "max_sharpe", "Bear": "min_vol_target_return", "Crisis": "min_vol"}
    for regime in objectives:
        w = opt.optimize_for_regime(regime, mu, cov, objectives, cfg)
        assert w.shape == (3,)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)


def test_unrecognized_objective_string_raises(cfg):
    mu, cov = _toy_mu_cov()
    with pytest.raises(ValueError):
        opt.optimize_for_regime("Weird", mu, cov, {"Weird": "not_a_real_objective"}, cfg)
