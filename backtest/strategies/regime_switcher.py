from ..core.features import build_features, odds_regime
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class RegimeSwitcher(StrategyBase):
    name = "regime_switcher"

    def on_backtest_start(self):
        self.win = {k: {"w": 0.0, "n": 0.0} for k in ("A_1p8", "B_2p2", "C_3p0", "D_5p0", "E_gt5")}

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        rg = odds_regime(selected.odds)
        d = self.win[rg]
        wr = (d["w"] + 1.0) / (d["n"] + 2.0)
        if wr >= 0.60:
            base = 95000
        elif wr >= 0.50:
            base = 80000
        elif wr >= 0.40:
            base = 60000
        else:
            base = 35000
        return base + 50000 * feat["ev"]

    def on_event_end(self, event, selected, won, pnl, state):
        rg = odds_regime(selected.odds)
        self.win[rg]["n"] += 1.0
        self.win[rg]["w"] += 1.0 if won else 0.0
