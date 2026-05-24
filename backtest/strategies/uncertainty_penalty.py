from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class UncertaintyPenalty(StrategyBase):
    name = "uncertainty_penalty"

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        base = 80000 + 120000 * feat["ev"]
        penalty = 1.0
        if feat["conf"] < 0.5:
            penalty *= 0.75
        if selected.odds > 3:
            penalty *= 0.8
        if selected.odds > 5:
            penalty *= 0.65
        if feat["ratio"] > 1.25:
            penalty *= 0.8
        if feat["ratio"] < 0.9 and feat["conf"] > 0.6:
            penalty *= 1.08
        return base * penalty
