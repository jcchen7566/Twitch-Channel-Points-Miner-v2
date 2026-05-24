from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class PercentileEvGate(StrategyBase):
    name = "percentile_ev_gate"

    def on_backtest_start(self):
        self.ev_hist = []

    def pick(self, event, state):
        return best_by_ev(event, state)

    def _q(self, p):
        if not self.ev_hist:
            return 0.0
        a = sorted(self.ev_hist)
        return a[int((len(a) - 1) * p)]

    def size(self, event, selected, state):
        ev = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))["ev"]
        q20 = self._q(0.2)
        q50 = self._q(0.5)
        q80 = self._q(0.8)
        if ev < q20:
            return 20000
        if ev < q50:
            return 45000
        if ev < q80:
            return 80000
        return 115000

    def on_event_end(self, event, selected, won, pnl, state):
        ev = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))["ev"]
        self.ev_hist.append(ev)
