import math
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase

class MLFeatureStrategy(StrategyBase):
    """
    A strategy that utilizes advanced mathematical feature engineering:
    - Z-score relative features
    - Interaction terms (Crowd x Liquidity)
    - Sigmoid confidence scaling
    """
    name = "ml_feature_optimized"

    def pick(self, event, state):
        best = None
        best_ev = -10**9
        for o in event.outcomes:
            # Use event-relative features (Z-scores)
            prior = state.prior_win_rate(event.streamer, o.odds)
            f = build_features(o, prior, event_outcomes=event.outcomes)
            
            # Adjusted probability based on relative crowd sentiment
            # If rel_ratio is high (crowd is heavily on one side), adjust p
            p_adj = f["p"]
            if f["rel_ratio"] > 1.5: p_adj *= 0.9  # Crowd herd bias penalty
            if f["rel_ratio"] < -1.5: p_adj *= 1.1 # Contrarian signal boost
            
            ev = p_adj * o.odds - 1.0
            if ev > best_ev:
                best_ev = ev
                best = o
        return best

    def size(self, event, selected, state):
        prior = state.prior_win_rate(event.streamer, selected.odds)
        f = build_features(selected, prior, event_outcomes=event.outcomes)
        
        # Mathematical Sizing Formula
        # base + (EV * multiplier) * confidence * liquidity_factor
        ev_component = max(0, f["ev"]) * 150000
        liq_factor = clamp(f["log_liq"] / 6.0, 0.5, 1.2)
        
        # Interaction effect: if liquidity is high and crowd is polarized, be more cautious
        if f["crowd_liq"] > 8.0 and abs(f["rel_ratio"]) > 1.0:
            liq_factor *= 0.8
            
        amount = (60000 + ev_component) * f["conf"] * liq_factor
        
        # Odds regime penalty
        if selected.odds > 4.0: amount *= 0.85
        if selected.odds > 7.0: amount *= 0.70
        
        return amount
