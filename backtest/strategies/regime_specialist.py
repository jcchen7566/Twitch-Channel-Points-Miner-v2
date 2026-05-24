from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class RegimeSpecialist(StrategyBase):
    name = "regime_specialist"

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        ev = feat["ev"]
        if selected.odds <= 1.8:
            return 90000 + 60000 * ev
        if selected.odds <= 2.2:
            return 80000 + 50000 * ev
        if selected.odds <= 3.0:
            return 65000 + 30000 * ev
        if selected.odds <= 5.0:
            return 45000 + 15000 * ev
        return 25000 + 5000 * ev
