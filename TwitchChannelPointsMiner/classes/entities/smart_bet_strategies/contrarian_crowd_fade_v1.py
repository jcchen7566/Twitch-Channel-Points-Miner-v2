from .base import SmartBetStrategyBase


class ContrarianCrowdFadeV1(SmartBetStrategyBase):
    """
    Deployed version of contrarian_crowd_fade.
    Fades crowd points-concentration when it deviates heavily from odds.
    Excellent for greentea.
    """
    name = "contrarian_crowd_fade_v1"

    def choose(self, bet_obj, details):
        best_index = 0
        best_score = -10**9
        
        for i, d in enumerate(details):
            odds = d["odds"]
            ev = d["ev"]
            outcome = bet_obj.outcomes[i]
            points_pct = float(outcome.get("odds_percentage") or 0.0)
            implied_p = (100.0 / odds) if odds > 0 else 100.0
            surprise = (points_pct - implied_p) / 100.0
            
            # Contrarian logic: penalize outcomes with high point surprise
            # (assumes crowd is often wrong on this streamer)
            score = ev - 0.25 * surprise
            
            if score > best_score:
                best_score = score
                best_index = i
                
        return best_index

    def size(self, bet_obj, balance, selected_detail):
        ev = selected_detail["ev"]
        amount = 40000 + 80000 * max(0, ev)
        
        amount = int(amount)
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "contrarian_crowd_fade_v1", "tier_fade_v1"
