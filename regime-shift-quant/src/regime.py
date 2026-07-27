"""Hidden Markov Model regime classifier.

Thin wrapper around `hmmlearn.hmm.GaussianHMM` that adds the two bits of
bookkeeping every user of an HMM for regime detection needs:

1. State-index -> human label mapping (HMM states are unordered integers;
   we map them to Bull/Bear/Crisis post-hoc by ranking average volatility).
2. A `fit_predict` path that is safe to call independently inside each
   walk-forward fold — the caller is responsible for only ever passing
   this class training-fold data at `.fit()` time (see backtest.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_LABELS_BY_VOL_RANK = ["Bull", "Bear", "Crisis"]  # low -> high volatility


@dataclass
class RegimeHMM:
    n_states: int = 3
    covariance_type: str = "diag"
    n_iter: int = 200
    random_state: int = 42
    vol_feature_for_labeling: str = "vol_21d"

    model: object = field(default=None, repr=False)
    state_to_label: Dict[int, str] = field(default_factory=dict)

    def _build_model(self):
        from hmmlearn import hmm  # lazy import — keeps src importable without hmmlearn installed

        return hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray) -> "RegimeHMM":
        """Fit on TRAINING DATA ONLY. Caller owns the train/test boundary."""
        self.model = self._build_model()
        self.model.fit(X)
        return self

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """Viterbi-decode the most likely state sequence for X.

        Safe to call on held-out data: this only uses the transition and
        emission parameters already learned in `.fit()`, not any
        information from X itself beyond the sequence of observations
        being decoded.
        """
        if self.model is None:
            raise RuntimeError("Call .fit() before .predict_states().")
        return self.model.predict(X)

    def learn_state_labels(self, feat_df: pd.DataFrame, state_col: str = "state") -> Dict[int, str]:
        """Rank states by mean volatility on the SAME data used to fit,
        producing a state-index -> {'Bull','Bear','Crisis'} mapping.

        Must be called with training-fold data (or, for a final
        full-sample diagnostic fit, the full sample) — never with a
        mapping learned on data the fold hasn't seen yet.
        """
        n = feat_df[state_col].nunique()
        state_vol = feat_df.groupby(state_col)[self.vol_feature_for_labeling].mean().sort_values()
        labels = DEFAULT_LABELS_BY_VOL_RANK
        if n > len(labels):
            labels = labels + [f"State{i}" for i in range(len(labels), n)]
        mapping = {state_idx: labels[i] for i, state_idx in enumerate(state_vol.index)}
        self.state_to_label = mapping
        return mapping

    def label(self, states: np.ndarray) -> List[str]:
        if not self.state_to_label:
            raise RuntimeError("Call .learn_state_labels() before .label().")
        return [self.state_to_label.get(s, f"State{s}") for s in states]

    @property
    def transition_matrix(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit yet.")
        return self.model.transmat_

    @property
    def means(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit yet.")
        return self.model.means_


def fit_and_label(
    feature_matrix: np.ndarray,
    raw_feat_df: pd.DataFrame,
    feature_columns: List[str],
    n_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 200,
    random_state: int = 42,
    vol_feature_for_labeling: str = "vol_21d",
) -> RegimeHMM:
    """Fit an HMM on `feature_matrix` and derive its Bull/Bear/Crisis labels.

    `raw_feat_df` must be row-aligned with `feature_matrix` and contain
    `vol_feature_for_labeling` so states can be ranked by volatility.
    Everything passed in here must come from the SAME fold (typically the
    training fold) to keep the labeling leakage-free.
    """
    rh = RegimeHMM(
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        vol_feature_for_labeling=vol_feature_for_labeling,
    )
    rh.fit(feature_matrix)
    states = rh.predict_states(feature_matrix)
    tmp = raw_feat_df.copy()
    tmp["state"] = states
    rh.learn_state_labels(tmp)
    return rh
