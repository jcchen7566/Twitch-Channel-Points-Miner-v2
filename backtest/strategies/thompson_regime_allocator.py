from ..core.features import odds_regime
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class ThompsonRegimeAllocator(StrategyBase):
    name = "thompson_regime_allocator"

    def on_backtest_start(self):
        self.posterior = {k: {"a": 1.0, "b": 1.0} for k in ("A_1p8", "B_2p2", "C_3p0", "D_5p0", "E_gt5")}

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        rg = odds_regime(selected.odds)
        post = self.posterior[rg]
        mean = post["a"] / (post["a"] + post["b"])
        return 20000 + 100000 * mean

    def on_event_end(self, event, selected, won, pnl, state):
        rg = odds_regime(selected.odds)
        post = self.posterior[rg]
        if won:
            post["a"] += 1.0
        else:
            post["b"] += 1.0
