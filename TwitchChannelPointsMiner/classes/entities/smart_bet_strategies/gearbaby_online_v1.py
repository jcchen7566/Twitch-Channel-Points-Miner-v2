from .base import SmartBetStrategyBase


class GearbabyOnlineV1(SmartBetStrategyBase):
    name = "gearbaby_online_v1"

    def choose(self, bet_obj, details):
        best_idx = 0
        best_score = -10**9
        for i, d in enumerate(details):
            # online_model-like deterministic proxy: EV core + confidence, penalize crowd-heavy side.
            score = d["ev"] + 0.10 * (d["confidence"] - 0.5) - 0.08 * max(0.0, d["ratio"] - 1.0)
            if d["odds"] > 4.5:
                score *= 0.85
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        amount = 70000 + 110000 * selected_detail["ev"]
        if selected_detail["confidence"] < 0.50:
            amount *= 0.85
        if selected_detail["ratio"] > 1.10:
            amount *= 0.85
        if selected_detail["odds"] > 4.5:
            amount *= 0.78
        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        return amount, amount / max(1, bet_obj.settings.base_bet_amount), "gearbaby_online_v1", "tier_gearbaby_v1"
