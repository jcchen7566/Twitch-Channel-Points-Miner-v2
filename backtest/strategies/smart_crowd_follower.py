import math
from collections import defaultdict
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase

class SmartCrowdFollower(StrategyBase):
    """
    Leans into streamers with high crowd-predictability (high AUC for users_pct).
    Uses relative surprise and crowd momentum.
    """
    name = "smart_crowd_follower"

    def on_backtest_start(self):
        # Tracking crowd accuracy per streamer
        self.crowd_score = defaultdict(lambda: {"correct": 0.0, "total": 0.0})

    def pick(self, event, state):
        best = None
        best_score = -10**9
        
        # Calculate crowd accuracy factor
        cs = self.crowd_score[event.streamer]
        acc = (cs["correct"] + 1.0) / (cs["total"] + 2.0)
        
        for o in event.outcomes:
            f = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
            
            # Prediction: Blend of EV and Crowd Sentiment adjusted by crowd accuracy
            # If crowd accuracy is high (like godjj), weigh surprise/ratio more.
            sentiment = f["rel_surprise"] + f["momentum"]
            score = (1.0 - acc) * f["ev"] + acc * sentiment
            
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        f = build_features(selected, state.prior_win_rate(event.streamer, selected.odds), event_outcomes=event.outcomes)
        cs = self.crowd_score[event.streamer]
        acc = (cs["correct"] + 1.0) / (cs["total"] + 2.0)
        
        # Aggressive sizing if crowd is known to be right
        multiplier = 40000 + 100000 * acc
        amount = 20000 + multiplier * max(0, f["ev"])
        
        if acc > 0.6: amount += 15000 # Confidence boost
        
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        # Update crowd accuracy tracking
        # Did the majority (points_pct) pick the winner?
        crowd_favorite = max(event.outcomes, key=lambda o: o.points_pct)
        correct = 1.0 if crowd_favorite.outcome_id == event.winner_outcome_id else 0.0
        
        self.crowd_score[event.streamer]["total"] += 1.0
        self.crowd_score[event.streamer]["correct"] += correct
