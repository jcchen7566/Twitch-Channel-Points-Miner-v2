# Backtest Architecture

## Principles
- Single input source: `backtest/core/log_parser.py` output only.
- Event-time simulation with no look-ahead.
- Strategy plugins must not modify engine internals.

## Modules
- `backtest/data_source.py`
  - Executes parser export.
  - Loads parser CSV into event/outcome objects.
- `backtest/engine.py`
  - Core simulation loop.
  - Risk clamp (min/max bet), PnL accounting, drawdown.
  - Metrics by streamer / odds regime / liquidity bucket.
- `backtest/strategy_base.py`
  - `StrategyBase` interface.
  - Online hierarchical prior state.
- `backtest/features.py`
  - Shared feature transforms from parser columns.
- `backtest/strategies/*.py`
  - Strategy implementations only.
- `backtest/strategies/registry.py`
  - Strategy group registration.
- `backtest/reporting.py`
  - Leaderboard and best-strategy breakdown export.
- `backtest/run.py`
  - Pipeline entrypoint.

## Add New Strategy
1. Create a class inheriting `StrategyBase` in `backtest/strategies/*.py`.
2. Implement `pick()` and `size()`.
3. If stateful, implement `on_backtest_start()` and/or `on_event_end()`.
4. Register in `backtest/strategies/registry.py`.

## Run
```bash
python3 backtest/run.py --db bet_logs.sqlite
```
