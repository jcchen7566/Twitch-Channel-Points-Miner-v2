from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class CrowdStructure(StrategyBase):
    name = "crowd_structure"

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        amount = 80000 + 90000 * feat["ev"]
        if feat["ratio"] < 0.9:
            amount += 15000
        elif feat["ratio"] > 1.2:
            amount -= 25000
        if selected.odds > 3.5:
            amount *= 0.82
        if selected.odds > 5.0:
            amount *= 0.72
        return amount
