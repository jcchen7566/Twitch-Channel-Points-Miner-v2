from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class DrawdownGuard(StrategyBase):
    name = "drawdown_guard"

    def on_backtest_start(self):
        self.equity = 0.0
        self.peak = 0.0

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        dd = self.peak - self.equity
        if dd < 100000:
            scale = 1.0
        elif dd < 300000:
            scale = 0.8
        elif dd < 600000:
            scale = 0.6
        else:
            scale = 0.45
        amount = (75000 + 90000 * feat["ev"]) * scale
        if selected.odds > 4.5:
            amount *= 0.85
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        self.equity += pnl
        if self.equity > self.peak:
            self.peak = self.equity
