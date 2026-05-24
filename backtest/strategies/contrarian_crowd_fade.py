import math
from collections import defaultdict
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase

class ContrarianCrowdFade(StrategyBase):
    """
    Fades the crowd for streamers where point concentration is a negative signal.
    Targets "Efficiency Divergence" cases.
    """
    name = "contrarian_crowd_fade"

    def on_backtest_start(self):
        # Tracking how often the heavy point favorite LOSES
        self.fade_score = defaultdict(lambda: {"losses": 0.0, "total": 0.0})

    def pick(self, event, state):
        best = None
        best_score = -10**9
        
        fs = self.fade_score[event.streamer]
        fade_prob = (fs["losses"] + 1.0) / (fs["total"] + 2.0)
        
        for o in event.outcomes:
            f = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
            
            # Contrarian logic: penalize outcomes with high point surprise if fade_prob is high
            contrarian_score = f["ev"] - fade_prob * f["rel_surprise"]
            
            if contrarian_score > best_score:
                best_score = contrarian_score
                best = o
        return best

    def size(self, event, selected, state):
        f = build_features(selected, state.prior_win_rate(event.streamer, selected.odds), event_outcomes=event.outcomes)
        fs = self.fade_score[event.streamer]
        fade_prob = (fs["losses"] + 1.0) / (fs["total"] + 2.0)
        
        # Base size adjusted by how contrarian we are
        amount = 40000 + 80000 * f["ev"]
        
        # If we are betting against a huge crowd surprise in a fade-heavy streamer, increase size
        if f["rel_surprise"] < -1.0 and fade_prob > 0.55:
            amount += 30000
            
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        crowd_favorite = max(event.outcomes, key=lambda o: o.points_pct)
        crowd_lost = 1.0 if crowd_favorite.outcome_id != event.winner_outcome_id else 0.0
        
        self.fade_score[event.streamer]["total"] += 1.0
        self.fade_score[event.streamer]["losses"] += crowd_lost
