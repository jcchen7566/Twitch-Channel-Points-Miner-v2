import math
from .base import SmartBetStrategyBase


class MLFeatureOptimizedV1(SmartBetStrategyBase):
    """
    Deployed version of ml_feature_optimized.
    Uses Z-score normalization and non-linear logic.
    """
    name = "ml_feature_optimized_v1"

    def choose(self, bet_obj, details):
        best_index = 0
        best_score = -10**9
        
        # 1. Calculate Z-scores for the event
        all_ratios = [d["ratio"] for d in details]
        mu_r = sum(all_ratios) / len(all_ratios)
        std_r = math.sqrt(sum((x - mu_r)**2 for x in all_ratios) / len(all_ratios)) + 1e-6
        
        for i, d in enumerate(details):
            odds = d["odds"]
            ev = d["ev"]
            ratio = d["ratio"]
            rel_ratio = (ratio - mu_r) / std_r
            
            p_adj = d["wr"]
            if rel_ratio > 1.5: p_adj *= 0.9  # Herd bias penalty
            if rel_ratio < -1.5: p_adj *= 1.1 # Contrarian boost
            
            score = p_adj * odds - 1.0
            
            if score > best_score:
                best_score = score
                best_index = i
                
        return best_index

    def size(self, bet_obj, balance, selected_detail):
        odds = selected_detail["odds"]
        ev = selected_detail["ev"]
        conf = selected_detail["confidence"]
        
        # Log-Liquidity factor proxy
        # Since we don't have total_points directly in detail, we use a default or derive it
        # But we can just use simple scaling
        
        ev_component = max(0, ev) * 150000
        amount = (60000 + ev_component) * conf
        
        if odds > 4.0: amount *= 0.85
        if odds > 7.0: amount *= 0.70
        
        amount = int(amount)
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "ml_feature_optimized_v1", "tier_ml_v1"
