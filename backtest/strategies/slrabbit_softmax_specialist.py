import math
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase

class SlrabbitSoftmaxSpecialist(StrategyBase):
    """
    Final Optimization for slrabbit99:
    1. Softmax Arm weights (Faster learning)
    2. Aggressive Sweet Spot Sizing (2.4 - 4.2)
    3. Selective Whale Fading (Only if extreme)
    """
    name = "slrabbit_softmax_specialist"

    def on_backtest_start(self):
        self.arms = {
            "LOW": {"n": 0, "avg": 0.0},
            "MID": {"n": 0, "avg": 0.0},
            "HIGH": {"n": 0, "avg": 0.0}
        }
        self.tau = 0.12 # More reactive

    def _arm(self, odds):
        if odds < 1.8: return "LOW"
        if odds <= 4.2: return "MID"
        return "HIGH"

    def pick(self, event, state):
        if event.streamer != "slrabbit99": return None

        best = None
        best_score = -10**9
        
        for o in event.outcomes:
            f = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
            arm = self._arm(o.odds)
            
            # 1. Softmax weight
            z = clamp(self.arms[arm]["avg"] / self.tau, -5, 5)
            softmax_weight = math.exp(z)
            
            # 2. Whale Awareness: Only penalize extreme concentration
            div = o.users_pct - o.points_pct
            whale_factor = 1.0
            if div < -18: whale_factor = 0.6
            elif div < -10: whale_factor = 0.85
            
            # 3. Sweet Spot: 2.4 - 4.2 is the core profit center
            sweet_spot = 1.4 if 2.4 <= o.odds <= 4.2 else 1.0
            
            score = (f["ev"] + 0.1) * softmax_weight * whale_factor * sweet_spot
            
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        f = build_features(selected, state.prior_win_rate(event.streamer, selected.odds), event_outcomes=event.outcomes)
        arm = self._arm(selected.odds)
        perf = self.arms[arm]["avg"]
        
        # Maximize size in the Sweet Spot
        if 2.4 <= selected.odds <= 4.2:
            base = 100000
        else:
            base = 50000
            
        amount = base + 150000 * max(0, f["ev"]) + 30000 * clamp(perf, -1, 1)
        
        # Cut size if extreme whale trap
        div = selected.users_pct - selected.points_pct
        if div < -12:
            amount *= 0.5
            
        return clamp(amount, 20000, 120000)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99": return
        arm = self._arm(selected.odds)
        a = self.arms[arm]
        a["n"] += 1
        a["avg"] += (pnl - a["avg"]) / a["n"]
