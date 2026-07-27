#!/usr/bin/env python3
"""Run the full Regime-Shift pipeline end to end.

    data -> features -> regime detection -> optimization -> backtest -> results

Usage
-----
    python scripts/run_pipeline.py --config config.yaml
    python scripts/run_pipeline.py --config config.yaml --start 2015-01-01 --force-refresh

All numbers referenced in the README's results section come from running
this script unmodified against the default config.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as data_mod
from src import features as feat_mod
from src import backtest as bt_mod
from src import metrics as metrics_mod
from src import optimization as opt_mod
from src import plotting as plot_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    p.add_argument("--start", default=None, help="Override data.start (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="Override data.end (YYYY-MM-DD).")
    p.add_argument("--force-refresh", action="store_true", help="Ignore the data cache and re-download.")
    p.add_argument("--no-plots", action="store_true", help="Skip generating PNG charts.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.start:
        cfg["data"]["start"] = args.start
    if args.end:
        cfg["data"]["end"] = args.end

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    logger.info("Step 1/5 — downloading and aligning price data...")
    prices = data_mod.load_price_panel(
        tickers=cfg["data"]["tickers"],
        vix_ticker=cfg["data"].get("vix_ticker"),
        start=cfg["data"]["start"],
        end=cfg["data"].get("end"),
        cache_dir=cfg["data"].get("cache_dir", "data_cache"),
        force_refresh=args.force_refresh,
    )
    asset_cols = list(cfg["data"]["tickers"].keys())
    asset_returns = data_mod.prices_to_simple_returns(prices, asset_cols)
    logger.info("Price panel: %d rows, %s -> %s", len(prices), prices.index.min().date(),
                prices.index.max().date())

    # ------------------------------------------------------------ features
    logger.info("Step 2/5 — engineering features...")
    feat_df = feat_mod.build_full_feature_matrix(
        asset_prices=prices,
        market_col="equity",
        vix=prices["vix"] if "vix" in prices.columns else None,
        momentum_windows=cfg["features"]["momentum_windows"],
        volatility_windows=cfg["features"]["volatility_windows"],
        vix_zscore_window=cfg["features"]["vix_zscore_window"],
    )

    combined = feat_df.join(asset_returns, rsuffix="_ret")
    combined = combined.dropna()
    feat_df = combined.loc[:, feat_df.columns]
    asset_returns = combined.loc[:, asset_cols]

    stress_periods = [("2020-02-15", "2020-04-15"), ("2022-01-01", "2022-10-31")]
    spikes = feat_mod.sanity_check_spikes(feat_df, "vol_21d", stress_periods)
    logger.info("Volatility sanity check around known stress periods:\n%s", spikes.to_string(index=False))

    # ------------------------------------------------------------- regime
    logger.info("Step 3/5 & 4/5 — walk-forward regime detection + regime-conditional optimization...")
    wf_cfg = bt_mod.WalkForwardConfig(
        initial_train_days=cfg["walk_forward"]["initial_train_days"],
        test_days=cfg["walk_forward"]["test_days"],
        min_holding_days=cfg["walk_forward"]["min_holding_days"],
        lookback_days=cfg["walk_forward"]["lookback_days"],
    )
    opt_cfg = opt_mod.OptimizationConfig(
        max_weight=cfg["optimization"]["max_weight"],
        min_weight=cfg["optimization"]["min_weight"],
        risk_free_rate_annual=cfg["optimization"]["risk_free_rate"],
        bear_target_annual_return=cfg["optimization"]["bear_target_annual_return"],
    )
    regime_cfg = {
        "n_states": cfg["regime"]["n_states"],
        "covariance_type": cfg["regime"]["covariance_type"],
        "n_iter": cfg["regime"]["n_iter"],
        "random_state": cfg["regime"]["random_state"],
        "vol_feature_for_labeling": "vol_21d",
    }

    result = bt_mod.run_walk_forward_backtest(
        feat_df=feat_df,
        asset_returns=asset_returns,
        feature_columns=cfg["regime"]["feature_columns"],
        wf_cfg=wf_cfg,
        regime_cfg=regime_cfg,
        opt_cfg=opt_cfg,
        objectives=cfg["optimization"]["objectives"],
        tx_cost_bps=cfg["costs"]["transaction_cost_bps"],
    )

    # ---------------------------------------------------------- benchmarks
    common_idx = result.returns.index
    bench_6040 = bt_mod.static_benchmark_returns(
        asset_returns.loc[common_idx], cfg["benchmarks"]["static_6040"]
    )
    bench_eq = bt_mod.static_benchmark_returns(
        asset_returns.loc[common_idx], cfg["benchmarks"]["equal_weight"]
    )

    # Also compute the dynamic strategy WITHOUT transaction costs, to show
    # the checklist's required "with vs without cost" comparison.
    result_no_cost = bt_mod.run_walk_forward_backtest(
        feat_df=feat_df,
        asset_returns=asset_returns,
        feature_columns=cfg["regime"]["feature_columns"],
        wf_cfg=wf_cfg,
        regime_cfg=regime_cfg,
        opt_cfg=opt_cfg,
        objectives=cfg["optimization"]["objectives"],
        tx_cost_bps=0.0,
    )

    # ------------------------------------------------------------- results
    logger.info("Step 5/5 — scoring strategies and writing outputs...")
    rebalances_per_year = 252 / max(1, cfg["walk_forward"]["min_holding_days"])

    summary = pd.DataFrame(
        {
            "Dynamic (net of costs)": metrics_mod.summarize(
                result.returns, result.weights_history, cfg["optimization"]["risk_free_rate"],
                rebalances_per_year=rebalances_per_year,
            ),
            "Dynamic (no costs)": metrics_mod.summarize(
                result_no_cost.returns, result_no_cost.weights_history,
                cfg["optimization"]["risk_free_rate"], rebalances_per_year=rebalances_per_year,
            ),
            "Static 60/40": metrics_mod.summarize(bench_6040, risk_free_annual=cfg["optimization"]["risk_free_rate"]),
            "Equal-Weight": metrics_mod.summarize(bench_eq, risk_free_annual=cfg["optimization"]["risk_free_rate"]),
        }
    ).T

    print("\n" + "=" * 78)
    print("PERFORMANCE SUMMARY")
    print("=" * 78)
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))
    print("=" * 78 + "\n")

    summary.to_csv(out_dir / "performance_summary.csv")
    result.returns.to_csv(out_dir / "dynamic_strategy_returns.csv", header=["return"])
    result.weights_history.to_csv(out_dir / "dynamic_strategy_weights.csv")
    result.regimes.to_csv(out_dir / "regime_labels.csv", header=["regime"])

    last_fold = max(result.fold_transition_matrices.keys())
    pd.DataFrame(
        result.fold_transition_matrices[last_fold],
        index=[result.fold_state_labels[last_fold][i] for i in range(len(result.fold_transition_matrices[last_fold]))],
        columns=[result.fold_state_labels[last_fold][i] for i in range(len(result.fold_transition_matrices[last_fold]))],
    ).to_csv(out_dir / "transition_matrix_last_fold.csv")

    if not args.no_plots:
        plot_mod.plot_price_with_regimes(
            prices.loc[result.regimes.index, "equity"], result.regimes,
            "Detected Regimes Overlaid on Equity Price (out-of-sample, walk-forward)",
            save_path=str(out_dir / "regimes_overlay.png"),
        )
        plot_mod.plot_equity_curves(
            {
                "Dynamic (net costs)": result.returns,
                "Dynamic (no costs)": result_no_cost.returns,
                "Static 60/40": bench_6040,
                "Equal-Weight": bench_eq,
            },
            "Out-of-Sample Equity Curves",
            save_path=str(out_dir / "equity_curves.png"),
        )
        plot_mod.plot_drawdown(
            result.returns, "Dynamic Strategy Drawdown", save_path=str(out_dir / "drawdown.png")
        )
        plot_mod.plot_transition_matrix(
            result.fold_transition_matrices[last_fold],
            result.fold_state_labels[last_fold],
            f"Transition Matrix (final walk-forward fold #{last_fold})",
            save_path=str(out_dir / "transition_matrix.png"),
        )

    logger.info("Done. Outputs written to %s/", out_dir)


if __name__ == "__main__":
    main()
