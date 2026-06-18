from .base import SmartBetStrategyBase


class SmartCrowdFollower(SmartBetStrategyBase):
    name = "smart_crowd_follower"

    def choose(self, bet_obj, details):
        best_idx = 0
        best_score = -10**9
        for i, d in enumerate(details):
            # crowd-follower with EV gate
            score = d["ev"] + 0.08 * min(1.5, d["ratio"]) + 0.06 * (d["confidence"] - 0.5)
            if d["odds"] > 5.0:
                score *= 0.85
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        amount = 70000 + 95000 * selected_detail["ev"]
        if selected_detail["ratio"] < 0.95:
            amount *= 0.90
        if selected_detail["odds"] > 5.0:
            amount *= 0.78
        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        return amount, amount / max(1, bet_obj.settings.base_bet_amount), "smart_crowd_follower", "tier_crowd_follow"
