# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

A program that looks at market data, decides whether the market is currently
**Bull**, **Bear**, or **Crisis**, and reshuffles a three-asset portfolio
(equity / bonds / gold) to match — using a Hidden Markov Model for regime
detection and convex optimization for the allocation, validated with a
walk-forward harness designed specifically to catch lookahead bias.

Built for the *Summer of Quant — Advanced Project* brief.

## Why this exists

Static 60/40-style portfolios don't know the world has changed. This engine
tries to detect the change (via an HMM on returns/volatility/momentum) and
respond to it (via a regime-specific convex optimization), while being
paranoid about the one thing that makes quant backtests lie: **letting the
model see the future during "testing."**

## Architecture

```
data.py          → yfinance download + caching + price/return alignment
features.py      → momentum & volatility features, built causally (rolling/diff only)
regime.py        → GaussianHMM wrapper: fit, Viterbi-decode, volatility-rank labeling
optimization.py  → cvxpy: max-Sharpe (Bull), min-vol-with-target-return (Bear), min-vol (Crisis)
backtest.py       → walk-forward folds + regime-conditional rebalancing + transaction costs
metrics.py        → Sharpe, Sortino, max drawdown, Calmar, turnover
plotting.py       → regime overlay, equity curves, drawdown, transition-matrix heatmap
scripts/run_pipeline.py → the one script that runs data → features → regime → optimization
                          → backtest → results, end to end
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate       # or your favourite env manager
pip install -r requirements.txt

python scripts/run_pipeline.py --config config.yaml
```

This downloads/caches price data (first run only), fits regimes fold-by-fold,
backtests the dynamic strategy against two static benchmarks, prints a
performance summary table, and writes CSVs + PNGs to `outputs/`.

Run the test suite:

```bash
pytest -v
```

Tests that need `hmmlearn` or `cvxpy` are automatically skipped
(`pytest.importorskip`) if those aren't installed, so `pytest` still gives a
useful signal on a partial environment.

## Key decisions (and why)

**Why 3 regimes?** The brief asks for Bull / Bear / Crisis specifically, and
economically these map to "calm uptrend," "steady decline," "violent,
high-vol drawdown." `n_components` is a modeling choice, not something the
HMM discovers on its own — 3 is small enough to stay interpretable and
resist overfitting to noise on a few thousand daily observations.

**Why these features (`log_ret`, `vol_21d`, `mom_21d`)?** They're the
minimum set that actually discriminates the three regimes: same-day
direction, a trailing-month realized-vol estimate (the main "is this a
crisis" signal), and a trailing-month momentum estimate (the main
"bull vs bear" signal). All three are built with `.rolling()` / `.diff()`,
which by construction only look backward — see `tests/test_features.py`,
which asserts that truncating the price series after day *t* never changes
the feature value computed at day *t* (the mechanical definition of "causal
feature").

**Why `covariance_type="diag"`?** Fewer parameters than `"full"`, lower
overfitting risk with a handful of features and a training fold of a few
hundred to a couple thousand days. Worth revisiting once you have more
features or more history.

**How lookahead bias is actually avoided (not just discussed):**
1. Walk-forward folds are strictly expanding-train / non-overlapping-test
   (`backtest.make_folds`) — a fold's training window ends exactly where its
   test window begins.
2. The `StandardScaler` and the `GaussianHMM` are refit from scratch inside
   every fold, on that fold's training data only (`backtest.run_walk_forward_backtest`).
   State→label mapping (which state is "Crisis") is also learned per-fold on
   training data only — it is never learned once on the full sample and then
   reused.
3. A day's portfolio weights are computed using a trailing return window that
   ends the day *before* the return they're applied to (`decision_i = day_i - 1`)
   — never using same-day or future returns to pick same-day weights.
4. `tests/test_backtest.py` encodes the fold-adjacency invariant as a
   regression test so a future refactor that reintroduces leakage fails CI,
   not just a manual code review.

**Rebalancing rule:** rebalance when the detected regime changes AND at
least `min_holding_days` (default 5) have passed since the last rebalance —
this exists purely to stop the strategy from whipsawing on single-day regime
flicker and racking up transaction costs for no edge.

**Documented simplification:** between rebalances, weights are held fixed
rather than drifting with relative asset performance (true buy-and-hold
drift tracking is a straightforward but bookkeeping-heavy extension — see
"Extending this project" below).

**Transaction costs:** 5–10bps per the brief; default is 7.5bps, charged as
`turnover_amount × bps/10,000` on rebalance days only. The pipeline runs the
identical backtest with `tx_cost_bps=0` alongside the real one specifically
so you can see the cost's effect, per the checklist requirement.

**Tickers:** defaults in `config.yaml` are NSE-flavoured — `^NSEI` (Nifty 50)
for equity, `GOLDBEES.NS` for gold, `LIQUIDBEES.NS` as a low-duration
bond/cash-like proxy (a true long-duration Indian G-Sec ETF ticker on
Yahoo Finance is inconsistent over time, so this is deliberately a stand-in
you're encouraged to swap for whatever bond-proxy ETF you trust), and
`^INDIAVIX` as the fear gauge.

## Troubleshooting

**`cvxpy.error.SolverError: The solver ECOS is not installed.`** — some
`cvxpy` installs (notably on very new Python versions, e.g. 3.14, where not
every solver has published wheels yet) don't ship every solver. `src/optimization.py`
no longer hardcodes a solver — it lets `cvxpy` auto-select from whatever's
installed, falling back through `CLARABEL` → `ECOS` → `SCS` → `OSQP` → `CVXOPT`.
If you still hit this, run `python -c "import cvxpy; print(cvxpy.installed_solvers())"`
to see what's actually available in your environment, and `pip install ecos`
(or `pip install clarabel`) to add one back.

**`ModuleNotFoundError: No module named 'yaml'`** — the PyPI package name is
`pyyaml`, not `yaml`; `pip install pyyaml` (already in `requirements.txt`).

## Reproducing results

```bash
python scripts/run_pipeline.py --config config.yaml --start 2012-01-01
```

Outputs land in `outputs/`:
- `performance_summary.csv` — Sharpe/Sortino/CAGR/MaxDD/Calmar/turnover for
  the dynamic strategy (with and without costs) vs. static 60/40 and
  equal-weight.
- `dynamic_strategy_returns.csv`, `dynamic_strategy_weights.csv`,
  `regime_labels.csv` — full daily detail.
- `transition_matrix_last_fold.csv` + `.png` — the regime persistence matrix
  from the most recent walk-forward fold.
- `regimes_overlay.png`, `equity_curves.png`, `drawdown.png`.

Numbers will vary with the date range you pull (`data.start`/`data.end` in
`config.yaml`) since yfinance always serves up-to-date history — that's
expected and is itself a demonstration that nothing here is hard-coded to a
specific historical run.

## Extending this project

- **Weight drift between rebalances:** currently weights are held flat;
  tracking actual portfolio drift day-to-day (each asset's weight moves with
  its realized return until the next rebalance) is a natural next step and
  would make the turnover/cost accounting slightly more realistic.
- **FRED macro features:** `config.yaml`/`src/data.py` are structured to make
  adding a macro data source (CPI, yield spreads) straightforward — pull it
  alongside VIX and add it to `regime.feature_columns`.
- **A supervised sanity-check classifier:** the project guide notebook
  sketches a small PyTorch feedforward net as an optional cross-check against
  the unsupervised HMM; not included here as a default dependency to keep
  the core pipeline lightweight, but `src/regime.py` is the natural place to
  add it.

## Tech stack

Python 3.10+ · NumPy · Pandas · SciPy · Matplotlib · yfinance · hmmlearn ·
CVXPY · scikit-learn

## License

MIT — see [LICENSE](LICENSE).
