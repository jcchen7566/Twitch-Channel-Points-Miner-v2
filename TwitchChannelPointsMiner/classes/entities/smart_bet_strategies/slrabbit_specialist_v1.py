from .base import SmartBetStrategyBase


class SlrabbitSpecialistV1(SmartBetStrategyBase):
    name = "slrabbit_specialist_v1"

    def choose(self, bet_obj, details):
        best_index = 0
        best_score = -10**9
        for i, d in enumerate(details):
            score = d["ev"] - 0.05 * max(0.0, d["ratio"] - 1.0)
            if d["odds"] > 3.5:
                score *= 0.80
            if score > best_score:
                best_score = score
                best_index = i
        return best_index

    def size(self, bet_obj, balance, selected_detail):
        amount = 70000 + 100000 * selected_detail["ev"]
        if selected_detail["odds"] > 3.5:
            amount *= 0.75
        if selected_detail["odds"] > 5.0:
            amount *= 0.70
        if selected_detail["ratio"] > 1.10:
            amount *= 0.85
        if selected_detail["confidence"] < 0.50:
            amount *= 0.85

        amount = int(amount)
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "slrabbit_specialist_v1", "tier_slrabbit_v1"
