from .base import SmartBetStrategyBase


class Fixed20k(SmartBetStrategyBase):
    name = "fixed_20k"

    def choose(self, bet_obj, details):
        return max(range(len(details)), key=lambda idx: details[idx]["ev"])

    def size(self, bet_obj, balance, selected_detail):
        amount = min(20000, bet_obj.settings.max_points, balance)
        amount = max(bet_obj.settings.min_bet_amount, int(amount))
        return amount, amount / max(1, bet_obj.settings.base_bet_amount), "fixed_20k", "tier_20k"
