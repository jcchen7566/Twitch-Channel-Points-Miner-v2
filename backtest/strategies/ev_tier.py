from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class EvTier(StrategyBase):
    name = "ev_tier"

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        e = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))["ev"]
        if e < 0:
            return 20000
        if e < 0.03:
            return 40000
        if e < 0.08:
            return 60000
        if e < 0.15:
            return 80000
        if e < 0.25:
            return 96000
        return 112000
