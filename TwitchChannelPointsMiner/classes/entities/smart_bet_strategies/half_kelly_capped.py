from .base import SmartBetStrategyBase


class HalfKellyCapped(SmartBetStrategyBase):
    name = "half_kelly_capped"

    def choose(self, bet_obj, details):
        return max(range(len(details)), key=lambda idx: details[idx]["ev"])

    def size(self, bet_obj, balance, selected_detail):
        odds = max(1.01, selected_detail["odds"])
        p = max(0.02, min(0.98, selected_detail["wr"]))
        b = odds - 1.0
        q = 1.0 - p
        k = max(0.0, (b * p - q) / b) * 0.5
        multiplier = 0.25 + min(1.15, k * 2.2)
        amount = int(bet_obj.settings.base_bet_amount * multiplier)
        if odds > 5.0:
            amount = int(amount * 0.75)
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        return amount, float(multiplier), "half_kelly", "tier_kelly"
