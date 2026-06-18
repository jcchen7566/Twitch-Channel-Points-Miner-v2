from .base import SmartBetStrategyBase


class EmpiricalBayesShrinkage(SmartBetStrategyBase):
    name = "empirical_bayes_shrinkage"

    def choose(self, bet_obj, details):
        best_idx = 0
        best_score = -10**9
        for i, d in enumerate(details):
            n = max(0.0, d.get("confidence", 0.5) * 25.0)
            lam = n / (n + 12.0)
            prior = 0.5
            p_shrink = lam * d["wr"] + (1.0 - lam) * prior
            score = p_shrink * d["odds"] - 1.0
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        n = max(0.0, selected_detail.get("confidence", 0.5) * 25.0)
        lam = n / (n + 12.0)
        prior = 0.5
        p_shrink = lam * selected_detail["wr"] + (1.0 - lam) * prior
        ev = p_shrink * selected_detail["odds"] - 1.0

        amount = 65000 + 85000 * ev
        if selected_detail["odds"] > 5.0:
            amount *= 0.75
        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        return amount, amount / max(1, bet_obj.settings.base_bet_amount), "empirical_bayes_shrinkage", "tier_ebs"
