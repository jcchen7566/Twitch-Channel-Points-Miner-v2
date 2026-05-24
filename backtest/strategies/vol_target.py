import math
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class VolTarget(StrategyBase):
    name = "vol_target"

    def on_backtest_start(self):
        self.pnl_hist = []

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        recent = self.pnl_hist[-25:]
        if len(recent) >= 8:
            mu = sum(recent) / len(recent)
            var = sum((x - mu) ** 2 for x in recent) / len(recent)
            vol = math.sqrt(max(0.0, var))
        else:
            vol = 0.0
        scale = 1.0 if vol <= 25000 else (25000.0 / vol)
        return 80000.0 * clamp(scale, 0.35, 1.2) + 70000.0 * feat["ev"]

    def on_event_end(self, event, selected, won, pnl, state):
        self.pnl_hist.append(pnl)
