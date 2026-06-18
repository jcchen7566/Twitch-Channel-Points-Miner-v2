from .base import SmartBetStrategyBase


class MlFeatureOptimized(SmartBetStrategyBase):
    name = "ml_feature_optimized"

    def choose(self, bet_obj, details):
        best_idx = 0
        best_score = -10**9
        for i, d in enumerate(details):
            # lightweight feature blend mirroring backtest intent
            score = (
                1.00 * d["ev"]
                + 0.10 * (d["wr"] - 0.5)
                + 0.08 * (d["confidence"] - 0.5)
                - 0.12 * max(0.0, d["ratio"] - 1.0)
            )
            if d["odds"] > 4.5:
                score *= 0.86
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        amount = 70000 + 100000 * selected_detail["ev"]
        amount += 15000 * (selected_detail["wr"] - 0.5)
        amount += 12000 * (selected_detail["confidence"] - 0.5)
        if selected_detail["ratio"] > 1.10:
            amount *= 0.82
        if selected_detail["odds"] > 5.0:
            amount *= 0.75
        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        return amount, amount / max(1, bet_obj.settings.base_bet_amount), "ml_feature_optimized", "tier_ml_opt"
