from .features import clamp, liquidity_bucket, odds_regime
from .types import BacktestConfig, BacktestResult, SliceStats


def _update_slice(target: dict[str, SliceStats], key: str, bet: float, gain: float, won: bool) -> None:
    s = target.get(key)
    if s is None:
        s = SliceStats()
        target[key] = s
    s.bet += bet
    s.gain += gain
    s.bets += 1
    if won:
        s.wins += 1
    else:
        s.losses += 1


def run_backtest(events: list, strategy, config: BacktestConfig) -> BacktestResult:
    from .strategy_base import OnlineStats

    state = OnlineStats()
    strategy.on_backtest_start()

    total_bet = 0.0
    total_gain = 0.0
    wins = 0
    losses = 0
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    skipped = 0

    by_streamer = {}
    by_regime = {}
    by_liquidity = {}

    for event in events:
        if not event.label_known:
            skipped += 1
            continue

        selected = strategy.pick(event, state)
        if selected is None:
            continue
            
        amount = clamp(strategy.size(event, selected, state), config.min_bet, config.max_bet)
        won = selected.outcome_id == event.winner_outcome_id
        gain = amount * ((selected.odds - 1.0) if won else -1.0)

        total_bet += amount
        total_gain += gain
        equity += gain
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
        wins += 1 if won else 0
        losses += 0 if won else 1

        _update_slice(by_streamer, event.streamer, amount, gain, won)
        _update_slice(by_regime, odds_regime(selected.odds), amount, gain, won)
        _update_slice(by_liquidity, liquidity_bucket(selected.total_points), amount, gain, won)

        strategy.on_event_end(event, selected, won, gain, state)
        state.update(event.streamer, selected.odds, won)

    n = wins + losses
    return BacktestResult(
        name=strategy.name,
        bets=n,
        total_bet=total_bet,
        total_gain=total_gain,
        roi=(total_gain / total_bet) if total_bet > 0 else 0.0,
        win_rate=(wins / n) if n > 0 else 0.0,
        avg_bet=(total_bet / n) if n > 0 else 0.0,
        max_drawdown=max_drawdown,
        skipped_unlabeled_events=skipped,
        by_streamer=by_streamer,
        by_regime=by_regime,
        by_liquidity=by_liquidity,
    )


def run_walk_forward(events: list, strategy_factory, config: BacktestConfig, train_size: int, test_size: int):
    folds = []
    start = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > len(events):
            break

        # currently strategies are online; fit phase reserved for future models.
        strategy = strategy_factory()
        _ = events[start:train_end]
        test_events = events[train_end:test_end]
        result = run_backtest(test_events, strategy, config)
        folds.append(result)
        start += test_size

    return folds
