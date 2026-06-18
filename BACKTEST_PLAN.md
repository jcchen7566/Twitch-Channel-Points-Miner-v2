# Backtesting Master Plan (Session Handoff)

## 1) Goal

Build a scalable, reproducible backtesting system for Twitch prediction betting that can support many strategy experiments (100+), while keeping data clean and model logic stable.

Primary optimization target:
- Per-streamer performance (not global one-size-fits-all).

Secondary target:
- Robustness across market regimes (odds, crowd behavior, liquidity).

---

## 2) Core Decisions Already Agreed

1. **Modeling scope is per-streamer by default.**
   - Each streamer should have separate strategy fit/evaluation.

2. **Do not rely on title text as primary classifier.**
   - Titles are noisy and unstable.
   - Use market structure features instead (odds, crowd, liquidity, etc).

3. **Use event-outcome snapshots (raw) for ML/backtesting.**
   - Avoid over-processed logs as primary training data.

4. **Single parser entrypoint.**
   - `scripts/log_parser.py` is the unified extraction interface.

5. **Current machine-readable storage = SQLite**
   - `bet_logs.sqlite`
   - Tables:
     - `bet_events` (event-level metadata + final result)
     - `bet_outcomes` (all outcomes per event with raw stats)

---

## 3) Current Data Contract (Do Not Break)

### `bet_events` includes:
- `event_id`, `created_at`, `resolved_at`
- `streamer`, `main_title`
- `status`, `status_reason`
- `balance_before`, `placed_amount`
- `selected_outcome_idx`, `selected_outcome_id`
- `winning_outcome_id`
- `result_type`, `points_won`, `points_gained`

### `bet_outcomes` includes:
- `event_id`, `outcome_idx`, `outcome_id`, `outcome_title`
- `outcome_users`, `outcome_points`, `outcome_odds`
- `outcome_users_percent`, `outcome_points_percent`
- `history_wins_weighted`, `history_total_weighted`
- `history_wins_raw`, `history_total_raw`
- `total_users`, `total_points`

### Notes
- One event has multiple rows in `bet_outcomes` (all choices).
- This is enough to backtest **selection among candidates** (not only sizing replay).

---

## 4) Logging / Data Policy Going Forward

1. Keep schema stable unless absolutely necessary.
2. If extension needed, prefer additive migration (new columns/tables), no breaking rename/remove.
3. Continue writing both:
   - weighted (decayed) history
   - raw (non-decayed) history
4. Missing `winning_outcome_id` for unresolved events is acceptable and expected.
5. Training data should typically filter to resolved rows:
   - `result_type IN ('WIN','LOSE','REFUND')`

---

## 5) Backtesting Architecture

## 5.1 Data layer
- Use `scripts/log_parser.py` to export normalized rows.
- Preferred training format: `jsonl` (one row per event-outcome snapshot).

## 5.2 Engine
- Event-time ordered simulation only (strictly no look-ahead).
- Group by event, evaluate candidate outcomes inside each event.
- Strategy output per event:
  - selected `outcome_id`
  - bet size

## 5.3 Strategy interface
- `fit(past_data)` optional
- `score(event_outcomes)` required
- `size(selected_outcome, context)` required

## 5.4 Experiment manager
- Runs many strategies over same period/splits.
- Outputs unified metrics table.

---

## 6) Modeling Approach

## 6.1 Base hierarchy (important)
1. Per-streamer base model.
2. Per-streamer + regime adjustment (odds regime).
3. Per-streamer + regime + crowd-ratio adjustment.

If sample size in a bucket is too small:
- back off to less granular level (hierarchical fallback).

## 6.2 Why not per-title/per-option model
- Too sparse and unstable.
- Overfitting risk is high.

---

## 7) Feature Set

## 7.1 Raw features
- `outcome_odds`
- `outcome_users_percent`
- `outcome_points_percent`
- `outcome_users`, `outcome_points`
- `total_users`, `total_points`
- `history_wins_weighted`, `history_total_weighted`
- `history_wins_raw`, `history_total_raw`

## 7.2 Derived features (recommended)
- `log_odds`
- `ratio = users% / points%`
- `users_points_gap = users% - points%`
- `odds_rank_in_event`
- `ratio_rank_in_event`
- `ratio_zscore_in_event`
- liquidity features:
  - `log_total_users`, `log_total_points`
  - liquidity bucket

## 7.3 Time-scale features
- `wr_weighted = wins_weighted / total_weighted`
- `wr_raw = wins_raw / total_raw`
- `wr_gap = wr_weighted - wr_raw`

## 7.4 Candidate-relative features (event-level)
- feature minus event mean
- feature minus best/second-best baseline
- `ev_gap_to_next` (if EV-style score used)

---

## 8) Labels / Targets

Per event-outcome row:
- `y = 1` if `outcome_id == winning_outcome_id`, else `0`.

This avoids dependence on option names and supports variable option text.

---

## 9) Validation Protocol

Use walk-forward / rolling time splits:
- Train window -> Test window, repeating chronologically.
- Never use future rows when fitting.

Minimum reliability checks:
- Per-streamer test sample count threshold.
- Regime-level minimum sample threshold.

---

## 10) Metrics

Required:
- Win rate
- ROI
- Total gain
- Avg bet size
- Max drawdown

Breakdowns:
- per streamer
- per odds regime
- per liquidity bucket

Reliability:
- number of bets/events per metric slice

---

## 11) Risk / Sizing Rules

Hard constraints:
- min bet = 20,000
- max bet = 120,000 (or current configured cap)

Sizing should be tested independently from selection:
1. selection-only benchmark (fixed bet)
2. sizing-on benchmark

This separates “pick quality” from “money management effect”.

---

## 12) Baseline Strategy Set for Large Runs

Keep these as reference baselines:
1. Fixed 20k
2. Fixed 80k
3. Simple EV tier
4. Half-Kelly (capped)
5. Crowd-structure heuristic
6. Regime specialist
7. Uncertainty-penalty sizing
8. Volatility target sizing
9. Quantile gate sizing
10. Online model baseline

All new strategies must be compared against this baseline pack.

---

## 13) Execution Plan (Next Session)

## Phase A: Data pipeline hardening
1. Validate parser export format for training.
2. Add deterministic event grouping utility.
3. Add integrity checks in engine (missing selected/winner, bad odds).

## Phase B: Engine + baselines
1. Implement generic backtest engine.
2. Implement baseline strategy pack.
3. Generate first benchmark report.

## Phase C: Per-streamer hierarchical modeling
1. Per-streamer model.
2. Regime fallback logic.
3. Add weighted/raw dual-history features.

## Phase D: Scale experiments
1. Batch runner for many strategies/hyperparams.
2. Report generation (leaderboard + reliability diagnostics).
3. Artifact/version tracking.

---

## 14) Non-Goals (for now)

- No deep NLP title modeling.
- No heavy black-box model before baseline stability.
- No schema churn.

---

## 15) Operational Commands (Current)

Example parser usage:

```bash
python3 scripts/log_parser.py --db bet_logs.sqlite --resolved-only --output /tmp/bet_rows.jsonl --format jsonl
python3 scripts/log_parser.py --db bet_logs.sqlite --resolved-only --stats
python3 scripts/log_parser.py --db bet_logs.sqlite --resolved-only --stats-by-streamer
```

---

## 16) Key Risks to Watch

1. Overfitting due to too many per-streamer sub-models with low sample.
2. Look-ahead leakage in feature construction.
3. Confusing selection quality with sizing effects.
4. Regime drift (short-term shift vs long-term historical bias).

Mitigation:
- walk-forward + sample thresholds + fallback hierarchy + ablation tests.

