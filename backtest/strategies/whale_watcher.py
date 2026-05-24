import math
from collections import defaultdict
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase

class WhaleWatcher(StrategyBase):
    """
    Differentiates between Retail (user count) and Whales (point count).
    For streamers like slrabbit99, data shows winners often have 
    HIGHER point concentration than user concentration (negative divergence).
    This strategy tracks who to trust: the whales or the retail crowd.
    """
    name = "whale_watcher"

    def on_backtest_start(self):
        # Tracking: bias > 0 means trust retail, bias < 0 means trust whales
        self.streamer_bias = defaultdict(float)
        self.counts = defaultdict(int)

    def pick(self, event, state):
        best = None
        best_score = -10**9
        bias = self.streamer_bias[event.streamer]
        
        for o in event.outcomes:
            f = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
            
            # Divergence: users_pct - points_pct
            # If bias is negative (trust whales), we like outcomes where points_pct > users_pct
            # If bias is positive (trust retail), we like outcomes where users_pct > points_pct
            divergence = o.users_pct - o.points_pct
            trust_score = -bias * divergence # Negative bias * divergence boosts target
            
            score = f["ev"] + 0.05 * trust_score
            
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        f = build_features(selected, state.prior_win_rate(event.streamer, selected.odds), event_outcomes=event.outcomes)
        bias = self.streamer_bias[event.streamer]
        divergence = selected.users_pct - selected.points_pct
        
        # Base size
        amount = 50000 + 80000 * f["ev"]
        
        # Boost size if current selection matches our trusted bias
        if (bias < 0 and divergence < -2.0) or (bias > 0 and divergence > 2.0):
            amount += 25000
            
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        # Update trust bias based on winner's divergence
        winner = next((o for o in event.outcomes if o.outcome_id == event.winner_outcome_id), None)
        if winner:
            div = winner.users_pct - winner.points_pct
            # Online moving average of divergence
            lr = 0.15
            self.streamer_bias[event.streamer] += lr * (div - self.streamer_bias[event.streamer])
            self.counts[event.streamer] += 1
