from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class QuantileGate(StrategyBase):
    name = "quantile_gate"

    def on_backtest_start(self):
        self.ev_seen = []

    def _q(self, p):
        if not self.ev_seen:
            return 0.0
        arr = sorted(self.ev_seen)
        return arr[int((len(arr) - 1) * p)]

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        e = feat["ev"]
        q20, q50, q80 = self._q(0.2), self._q(0.5), self._q(0.8)
        if e < q20:
            return 20000
        if e < q50:
            return 40000
        if e < q80:
            return 80000
        return 112000

    def on_event_end(self, event, selected, won, pnl, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        self.ev_seen.append(feat["ev"])
