import math
from ..core.features import build_features
from ..core.strategy_base import StrategyBase
from .common import best_by_ev


class SoftmaxBandit(StrategyBase):
    name = "softmax_bandit"

    def on_backtest_start(self):
        self.arms = {k: {"n": 0, "avg": 0.0} for k in ("A", "B", "C", "D", "E")}

    def _arm(self, odds):
        if odds <= 1.8: return "A"
        if odds <= 2.2: return "B"
        if odds <= 3.0: return "C"
        if odds <= 5.0: return "D"
        return "E"

    def pick(self, event, state):
        return best_by_ev(event, state)

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        tau = 0.15
        vals = [self.arms[k]["avg"] for k in ("A", "B", "C", "D", "E")]
        mx = max(vals)
        exps = [math.exp((v - mx) / tau) for v in vals]
        z = sum(exps)
        probs = {k: exps[i] / z for i, k in enumerate(("A", "B", "C", "D", "E"))}
        weight = probs[self._arm(selected.odds)]
        return 20000 + 100000 * weight + 50000 * max(-0.2, min(0.4, feat["ev"]))

    def on_event_end(self, event, selected, won, pnl, state):
        arm = self._arm(selected.odds)
        a = self.arms[arm]
        a["n"] += 1
        a["avg"] += (pnl - a["avg"]) / a["n"]
