import math
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase

class GoldenRegimeContrarian(StrategyBase):
    """
    Renamed from slrabbit_specialist.
    Identifies market inefficiencies in specific odds regimes and 
    fades the crowd point-concentration.
    """
    name = "golden_regime_contrarian"

    def pick(self, event, state):
        best = None
        best_score = -10**9
        
        for o in event.outcomes:
            f = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
            score = f["ev"]
            
            # 1. Odds Sweet Spot (Generic Golden Regime)
            if 1.8 <= o.odds <= 3.2:
                score += 0.15
            elif o.odds > 4.5:
                score -= 0.25
            
            # 2. Point Efficiency Fade
            if f["efficiency"] < -0.05:
                score += 0.08
            
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        f = build_features(selected, state.prior_win_rate(event.streamer, selected.odds), event_outcomes=event.outcomes)
        
        # Aggressive sizing in the golden regime
        if 1.9 <= selected.odds <= 2.9:
            amount = 110000 + 10000 * f["ev"]
        elif 1.6 <= selected.odds <= 3.5:
            amount = 85000 + 20000 * f["ev"]
        else:
            amount = 25000
            
        if f["efficiency"] > 0.1:
            amount *= 0.75
        
        return amount
