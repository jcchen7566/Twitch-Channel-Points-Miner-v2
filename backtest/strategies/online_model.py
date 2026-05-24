import math

from ..core.features import build_features
from ..core.strategy_base import StrategyBase


class OnlineModel(StrategyBase):
    name = "online_model"

    def __init__(self):
        self.w = [0.0] * 6
        self.lr = 0.03
        self.reg = 1e-4

    def _feat(self, selected, base):
        return [1.0, base["p"], math.log(max(1.01, selected.odds)), base["ratio"], base["conf"], base["ev"]]

    def _pred(self, selected, base):
        z = sum(a * b for a, b in zip(self.w, self._feat(selected, base)))
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def pick(self, event, state):
        best = None
        best_ev = -10**9
        for o in event.outcomes:
            base = build_features(o, state.prior_win_rate(event.streamer, o.odds))
            p = max(0.02, min(0.98, self._pred(o, base)))
            ev = p * o.odds - 1.0
            if ev > best_ev:
                best_ev = ev
                best = o
        return best

    def size(self, event, selected, state):
        base = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        p = max(0.02, min(0.98, self._pred(selected, base)))
        ev = p * selected.odds - 1.0
        amount = 80000 + 140000 * ev
        if p < 0.45:
            amount *= 0.8
        if selected.odds > 5.0:
            amount *= 0.7
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        base = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        x = self._feat(selected, base)
        p = max(0.02, min(0.98, self._pred(selected, base)))
        y = 1.0 if won else 0.0
        g = p - y
        for i in range(len(self.w)):
            self.w[i] -= self.lr * (g * x[i] + self.reg * self.w[i])
