# Developer & Agent Guidelines (AGENTS.md)

This document outlines the core architecture, data contracts, and design philosophies of the backtesting system. **All future agents and developers must strictly follow these rules.**

---

## 1. Directory Structure Rules

The `backtest` directory must remain clean, modular, and minimal. Do not dump scratch scripts or duplicate runners into it.

```
backtest/
├── run.py                 # The SINGLE, unified entrypoint for running backtests.
├── __init__.py
├── core/                  # Core simulation and data processing modules.
│   ├── __init__.py
│   ├── data_source.py     # CSV loading & event reconstruction.
│   ├── engine.py          # Sequential time-ordered simulation engine.
│   ├── features.py        # Z-score normalization & market indicators.
│   ├── reporting.py       # Leaderboard printer (grouped per streamer).
│   ├── strategy_base.py   # Base class for strategies & online stats tracking.
│   ├── title_clustering.py# Natural language title parsing.
│   └── types.py           # Dataclasses.
└── strategies/            # Strategy plugins (registry dynamically discovers these).
    ├── __init__.py
    ├── registry.py        # Scans strategies folder and loads factories.
    ├── common.py          # Common utility logic (e.g. best_by_ev).
    └── [strategies].py    # 30+ separate strategy implementations (e.g., omega_hybrid.py).
```

> [!IMPORTANT]
> **No Duplicated Runner Scripts**: Delete any standalone scripts like `run_backtest_from_log_parser.py`, `run_all.py`, or `optimize_omega.py` if they are created during testing. The only runner allowed is `backtest/run.py` (run via `python3 -m backtest.run`).
>
> **No Temporary/CSV Files in Project**: Do not save intermediate CSV, JSON, or log files inside the `backtest/` folder or anywhere in the project workspace. Any required temporary files must be created under `/tmp` (e.g., using python's `tempfile.NamedTemporaryFile`) and must be strictly cleaned up/deleted once execution finishes.

---

## 2. Core Backtesting Principles

1. **One Source of Truth**:
   - Always retrieve data by calling `scripts/log_parser.py` (which queries `bet_logs.sqlite`). Do not write direct sqlite queries in the backtesting runner or bypass the parser script.
2. **Per-Streamer Optimization**:
   - Do not optimize for global aggregate metrics alone. The backtesting engine must output and rank strategy performance **per streamer**.
   - The ultimate goal of backtesting is to identify and configure the **best strategy tailored for each streamer**, rather than using a single global strategy.
3. **No Look-Ahead Bias**:
   - The simulation must strictly process events sequentially by their timestamp. Never use future information in features, priors, or online training weights.

---


## Runtime Learning Strategy Rule

For all **learn-based runtime strategies** (e.g. bandit / online-learning style), implementation must follow this single-path rule:

- Runtime decision-time learning state must be rebuilt from `bet_logs.sqlite` historical resolved records **on each decision**.
- Do not rely on in-memory `on_event_end` updates as the primary runtime state path.
- Do not maintain dual learning paths (preload memory path + DB path) in runtime strategy implementations.
- Backtest and runtime must keep parameter parity (same core scoring/sizing parameters) unless a documented experiment explicitly states otherwise.

Rationale: this ensures restart safety, deterministic reproducibility, and avoids divergence between runtime and backtest behavior.

## 3. Market Inefficiencies & Strategy Insights

When designing or tuning strategies, respect these robust empirical facts from the prediction database:

* **High Liquidity is Too Efficient**:
   * Streamers/events with `total_points > 2,000,000` (Liquidity `L3_high`) have extremely efficient pricing. The average historical ROI is negative (**-2.2%**). Keep bets highly conservative or avoid high liquidity events entirely.
* **Low/Medium Liquidity holds the Edge**:
   * Inefficient, emotional fan betting in low liquidity (`<500k` points) and medium liquidity (`500k - 2M` points) streams yields extremely high ROIs (**+237%** and **+154%** respectively). Leaning into these is where the edge lies.
* **Penalize Favorites**:
   * Low-odds choices (`odds <= 1.8`) have a negative historical ROI (**-3.4%**). Avoid or downsize heavily on low-odds favorites.
* **Smart Money Flow (Whales)**:
   * When `users_pct / points_pct < 0.9` (ratio of crowd count to points count), it signifies that big whales are backing the outcome while the retail crowd is not. This is a very strong winning indicator.
* **Empirical Bayes Shrinkage**:
   * Because streamer sample sizes are often tiny (e.g. 1 to 15 events), raw online probability estimates overfit instantly. Always shrink raw streamer win rates towards the global regime priors using a shrinkage factor: `lam = n / (n + 12.0)`.
