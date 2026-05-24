import math
from .base import SmartBetStrategyBase


class SoftmaxBanditV1(SmartBetStrategyBase):
    """
    Production version of SoftmaxBandit.
    Learns performance across different odds arms online.
    Extremely robust and resistant to overfitting due to low complexity.
    """
    name = "softmax_bandit_v1"

    def __init__(self):
        # Arms: A(<=1.8), B(<=2.2), C(<=3.0), D(<=5.0), E(>5.0)
        # Using a shared state would be better, but for now we follow the backtest structure
        # In production, this state is maintained in memory during the process lifecycle.
        self.arms = {k: {"n": 0, "avg": 0.0} for k in ("A", "B", "C", "D", "E")}
        self.tau = 0.15

    def _get_arm_key(self, odds):
        if odds <= 1.8: return "A"
        if odds <= 2.2: return "B"
        if odds <= 3.0: return "C"
        if odds <= 5.0: return "D"
        return "E"

    def choose(self, bet_obj, details):
        # Softmax Bandit picks the best EV choice as base
        # But its sizing logic is where the 'learning' happens.
        # For selection, we follow the most rational choice.
        best_index = 0
        best_ev = -10**9
        for i, d in enumerate(details):
            if d["ev"] > best_ev:
                best_ev = d["ev"]
                best_index = i
        return best_index

    def size(self, bet_obj, balance, selected_detail):
        odds = selected_detail["odds"]
        ev = selected_detail["ev"]
        
        arm = self._get_arm_key(odds)
        
        # Calculate probabilities from Softmax
        vals = [self.arms[k]["avg"] for k in ("A", "B", "C", "D", "E")]
        mx = max(vals)
        
        # Prevent overflow and ensure stability
        exps = [math.exp(max(-10, min(10, (v - mx) / self.tau))) for v in vals]
        z = sum(exps)
        probs = {k: exps[i] / z for i, k in enumerate(("A", "B", "C", "D", "E"))}
        
        weight = probs[arm]
        
        # Logic: Base 20k + up to 100k based on historical performance of this odds arm
        # plus a kicker for current event EV
        amount = 20000 + 100000 * weight + 50000 * max(-0.2, min(0.4, ev))
        
        amount = int(amount)
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), f"softmax_{arm}", "tier_softmax"

    def on_event_end(self, event, selected, won, pnl, state):
        # NOTE: The production framework needs to call this!
        # Assuming the caller passes the outcome details
        # For current production Miner, we might need a bridge to update this.
        # However, for the strategy object itself, we implement the update.
        odds = getattr(selected, 'odds', 1.0)
        arm = self._get_arm_key(odds)
        a = self.arms[arm]
        a["n"] += 1
        a["avg"] += (pnl - a["avg"]) / a["n"]
