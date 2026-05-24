import math
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase
from ..core.title_clustering import cluster_title

class SlrabbitSpecialistV2(StrategyBase):
    """
    Highly specialized strategy for slrabbit99 based on data analysis:
    1. Fade Whales: If (points_pct - users_pct) > 10%, it's a trap.
    2. Odds Sweet Spot: High conviction on odds between 2.2 and 4.2.
    3. Regime Bias: Aggressive on Variety/Chatting, cautious on GamePlay.
    """
    name = "slrabbit_specialist_v2"

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None 

        best = None
        best_score = -10**9
        
        title_regime = cluster_title(event.main_title)
        
        for o in event.outcomes:
            f = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
            
            # Feature: Whale Trap (users_minus_points)
            div = o.users_pct - o.points_pct
            
            # If points concentration is high (div < -2), it's risky
            whale_trap_penalty = 0.0
            if div < -10: whale_trap_penalty = -0.8 # Ultra trap
            elif div < -2: whale_trap_penalty = -0.4 # Medium trap
            elif div > 10: whale_trap_penalty = 0.2  # Crowd support is good here
            
            # Feature: Odds Sweet Spot (2.4 to 4.5 is the gold mine)
            odds_score = 0.0
            if 2.4 <= o.odds <= 4.5:
                odds_score = 0.6
            elif o.odds < 1.4:
                odds_score = -0.2
            elif o.odds > 6.0:
                odds_score = -0.5
                
            # Regime Bonus
            regime_bonus = 0.2 if title_regime in ["REGIME_CHAT", "REGIME_VARIETY"] else 0.0
            
            # Combined score
            score = f["ev"] + whale_trap_penalty + odds_score + regime_bonus
            
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99": return 20000
        
        # Adaptive Sizing for slrabbit99
        odds = selected.odds
        div = selected.users_pct - selected.points_pct
        
        base = 60000
        
        # Aggressive in sweet spot
        if 2.4 <= odds <= 4.5:
            base = 120000
        elif odds < 1.6:
            base = 40000
            
        # Cut size if whale trap detected
        if div < -3:
            base *= 0.3
            
        # Boost if regime is variety
        title_regime = cluster_title(event.main_title)
        if title_regime in ["REGIME_CHAT", "REGIME_VARIETY"]:
            base *= 1.3
            
        return clamp(base, 20000, 120000)
