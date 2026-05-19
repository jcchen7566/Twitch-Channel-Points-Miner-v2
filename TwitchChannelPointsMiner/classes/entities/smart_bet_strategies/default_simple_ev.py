from .base import SmartBetStrategyBase


class DefaultSimpleEv(SmartBetStrategyBase):
    name = "default_simple_ev"

    def choose(self, bet_obj, details):
        return max(range(len(details)), key=lambda idx: details[idx]["ev"])

    def size(self, bet_obj, balance, selected_detail):
        if not bet_obj.settings.use_ev_sizing:
            base_amount = bet_obj.settings.base_bet_amount
            amount = int(max(bet_obj.settings.min_bet_amount, min(base_amount, bet_obj.settings.max_points, balance)))
            return amount, 1.0, "legacy_fixed", "tier_legacy"

        ev_adj = selected_detail["ev"]
        base = bet_obj.settings.base_bet_amount
        if ev_adj < 0.0:
            multiplier = 0.25
            reason = "negative_ev"
            tier = "tier_20k"
        else:
            boosted = min(0.50, ev_adj)
            multiplier = 1.00 + boosted
            reason = "positive_ev"
            if multiplier < 1.15:
                tier = "tier_80k"
            elif multiplier < 1.30:
                tier = "tier_96k"
            else:
                tier = "tier_112k"

        amount = int(base * multiplier)
        amount = max(20000, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        return amount, float(multiplier), reason, tier
