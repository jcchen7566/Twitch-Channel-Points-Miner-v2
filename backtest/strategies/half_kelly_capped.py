from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class HalfKellyCapped(StrategyBase):
    name = "half_kelly_capped"

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        p = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))["p"]
        b = max(0.01, selected.odds - 1.0)
        q = 1.0 - p
        k = max(0.0, (b * p - q) / b) * 0.5
        amount = 20000 + 100000 * min(1.0, k * 2.2)
        if selected.odds > 5.0:
            amount *= 0.75
        return amount
