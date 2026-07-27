import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("hmmlearn", reason="hmmlearn not installed")

from src import regime as regime_mod


def _synthetic_regime_data(n=600, seed=7):
    """Three clearly-separated Gaussian blocks in sequence -> an HMM should
    recover 3 states whose average 'vol' feature ranks Bull < Bear < Crisis.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)

    calm = rng.normal(loc=[0.001, 0.005, 0.01], scale=[0.0005, 0.001, 0.002], size=(n // 3, 3))
    falling = rng.normal(loc=[-0.001, 0.015, -0.01], scale=[0.0007, 0.002, 0.003], size=(n // 3, 3))
    crisis = rng.normal(loc=[-0.004, 0.04, -0.03], scale=[0.001, 0.005, 0.004], size=n - 2 * (n // 3))

    X = np.vstack([calm, falling, crisis])
    df = pd.DataFrame(X, columns=["log_ret", "vol_21d", "mom_21d"], index=idx)
    return df


def test_fit_and_label_produces_three_distinct_labels():
    df = _synthetic_regime_data()
    X = df[["log_ret", "vol_21d", "mom_21d"]].values
    rh = regime_mod.fit_and_label(
        feature_matrix=X,
        raw_feat_df=df,
        feature_columns=["log_ret", "vol_21d", "mom_21d"],
        n_states=3,
        n_iter=100,
    )
    assert set(rh.state_to_label.values()) == {"Bull", "Bear", "Crisis"}
    assert rh.transition_matrix.shape == (3, 3)
    # Rows of a transition matrix must sum to 1.
    np.testing.assert_allclose(rh.transition_matrix.sum(axis=1), np.ones(3), atol=1e-6)


def test_labeling_ranks_states_by_mean_volatility():
    df = _synthetic_regime_data()
    X = df[["log_ret", "vol_21d", "mom_21d"]].values
    rh = regime_mod.fit_and_label(
        feature_matrix=X,
        raw_feat_df=df,
        feature_columns=["log_ret", "vol_21d", "mom_21d"],
        n_states=3,
        n_iter=100,
    )
    states = rh.predict_states(X)
    tmp = df.copy()
    tmp["state"] = states
    tmp["label"] = rh.label(states)
    vol_by_label = tmp.groupby("label")["vol_21d"].mean()
    assert vol_by_label["Bull"] < vol_by_label["Bear"] < vol_by_label["Crisis"]
