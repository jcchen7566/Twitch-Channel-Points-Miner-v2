from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class Fixed20k(StrategyBase):
    name = "fixed_20k"

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        return 20000.0
