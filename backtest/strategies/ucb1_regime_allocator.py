import math

from ..core.features import odds_regime
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class UCB1RegimeAllocator(StrategyBase):
    name = "ucb1_regime_allocator"

    def on_backtest_start(self):
        self.n_total = 0
        self.arms = {k: {"n": 0, "avg": 0.0} for k in ("A_1p8", "B_2p2", "C_3p0", "D_5p0", "E_gt5")}

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        rg = odds_regime(selected.odds)
        arm = self.arms[rg]
        if arm["n"] == 0:
            ucb = 1.0
        else:
            ucb = arm["avg"] + math.sqrt(2.0 * math.log(max(2, self.n_total)) / arm["n"])
        weight = max(0.0, min(1.0, 0.5 + 0.35 * ucb))
        return 20000 + 100000 * weight

    def on_event_end(self, event, selected, won, pnl, state):
        rg = odds_regime(selected.odds)
        arm = self.arms[rg]
        self.n_total += 1
        arm["n"] += 1
        reward = 1.0 if won else 0.0
        arm["avg"] += (reward - arm["avg"]) / arm["n"]
