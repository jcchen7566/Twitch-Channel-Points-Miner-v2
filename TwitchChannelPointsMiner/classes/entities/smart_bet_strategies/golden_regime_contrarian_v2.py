from .base import SmartBetStrategyBase


class GoldenRegimeContrarianV2(SmartBetStrategyBase):
    """
    Deployed version of the backtested GoldenRegimeContrarian strategy.
    Optimized for slrabbit99 and other streamers with Underdog Bias.
    """
    name = "golden_regime_contrarian_v2"

    def choose(self, bet_obj, details):
        best_index = 0
        best_score = -10**9
        
        for i, d in enumerate(details):
            odds = d["odds"]
            ev = d["ev"]
            
            # Efficiency Proxy: (Points % - Implied Prob)
            # odds_percentage is usually the points % on Twitch
            outcome = bet_obj.outcomes[i]
            points_pct = float(outcome.get("odds_percentage") or 0.0)
            implied_p = (100.0 / odds) if odds > 0 else 100.0
            efficiency = (points_pct - implied_p) / 100.0
            
            score = ev
            
            # 1. Odds Sweet Spot (1.8 - 3.2)
            if 1.8 <= odds <= 3.2:
                score += 0.15
            elif odds > 4.5:
                score -= 0.25
                
            # 2. Point Efficiency Fade (Negative surprise boost)
            if efficiency < -0.05:
                score += 0.08
            
            if score > best_score:
                best_score = score
                best_index = i
                
        return best_index

    def size(self, bet_obj, balance, selected_detail):
        odds = selected_detail["odds"]
        ev = selected_detail["ev"]
        
        # Sizing logic from backtest
        if 1.9 <= odds <= 2.9:
            amount = 110000 + 10000 * ev
        elif 1.6 <= odds <= 3.5:
            amount = 85000 + 20000 * ev
        else:
            amount = 25000
            
        # Risk adjustment
        # Use the outcome data from bet_obj via the decision choice index
        idx = bet_obj.decision["choice"]
        outcome = bet_obj.outcomes[idx]
        points_pct = float(outcome.get("odds_percentage") or 0.0)
        implied_p = (100.0 / odds) if odds > 0 else 100.0
        efficiency = (points_pct - implied_p) / 100.0
        
        if efficiency > 0.1:
            amount *= 0.75
            
        # Safeguards
        amount = int(amount)
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "golden_regime_contrarian_v2", "tier_golden_v2"
