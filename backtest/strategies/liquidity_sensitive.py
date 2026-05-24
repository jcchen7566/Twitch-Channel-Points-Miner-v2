import math

from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class LiquiditySensitive(StrategyBase):
    name = "liquidity_sensitive"

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        liq = max(1.0, selected.total_points)
        liq_factor = max(0.65, min(1.25, math.log10(liq) / 6.0))
        amount = (70000 + 85000 * feat["ev"]) * liq_factor
        if selected.odds > 5.0:
            amount *= 0.78
        return amount
