"""Plotting helpers for the regime classifier and backtest results."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

REGIME_COLORS = {"Bull": "#2ecc71", "Bear": "#e67e22", "Crisis": "#e74c3c"}


def plot_price_with_regimes(price: pd.Series, regimes: pd.Series, title: str,
                             save_path: Optional[str] = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    common_idx = price.index.intersection(regimes.index)
    price = price.loc[common_idx]
    regimes = regimes.loc[common_idx]

    for label, color in REGIME_COLORS.items():
        mask = regimes == label
        if mask.any():
            ax.scatter(price.index[mask], price[mask], s=6, color=color, label=label)
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_equity_curves(returns_dict: Dict[str, pd.Series], title: str,
                        save_path: Optional[str] = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    for label, rets in returns_dict.items():
        equity = (1 + rets.fillna(0)).cumprod()
        ax.plot(equity.index, equity.values, label=label, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Growth of ₹1")
    ax.legend()
    fig.autofmt_xdate()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_transition_matrix(transmat: np.ndarray, labels: Dict[int, str], title: str,
                            save_path: Optional[str] = None):
    import matplotlib.pyplot as plt

    n = transmat.shape[0]
    tick_labels = [labels.get(i, str(i)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(transmat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{transmat[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(returns: pd.Series, title: str, save_path: Optional[str] = None):
    import matplotlib.pyplot as plt

    equity = (1 + returns.fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#e74c3c", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
