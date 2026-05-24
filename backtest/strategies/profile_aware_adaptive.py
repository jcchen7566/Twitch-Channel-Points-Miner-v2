import math
from collections import defaultdict
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase

class ProfileAwareAdaptive(StrategyBase):
    """
    New Optimized Algorithm:
    Learns three distinct 'Alphas' per streamer:
    1. EV-Follower Alpha: Does a higher theoretical EV actually lead to wins?
    2. Whale-Follower Alpha: Does point concentration (low ratio) predict the winner?
    3. Contrarian Alpha: Does the crowd favorite (high points_pct) usually lose?
    """
    name = "profile_aware_adaptive"

    def on_backtest_start(self):
        # Per-streamer profile state
        self.profiles = defaultdict(lambda: {
            "ev_alpha": 0.5,
            "whale_alpha": 0.3,
            "crowd_alpha": 0.0,
            "wins": 0,
            "total": 0,
            "n_events": 0,
            "reliability": 0.1
        })
        self.lr = 0.2 # Faster learning

    def pick(self, event, state):
        p = self.profiles[event.streamer]
        best = None
        best_score = -10**9
        
        for o in event.outcomes:
            f = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
            
            # Blend signals
            ev_signal = f["ev"] * p["ev_alpha"]
            # Whale signal: use users_minus_points directly
            whale_signal = (o.points_pct - o.users_pct) * p["whale_alpha"]
            crowd_signal = f["efficiency"] * p["crowd_alpha"]
            
            score = ev_signal + 0.1 * whale_signal + 0.05 * crowd_signal
            
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        p = self.profiles[event.streamer]
        f = build_features(selected, state.prior_win_rate(event.streamer, selected.odds), event_outcomes=event.outcomes)
        
        # Adaptive Kelly-ish sizing
        wr = (p["wins"] + 1.0) / (p["total"] + 2.0)
        conf = min(1.0, p["n_events"] / 5.0)
        
        # Base amount scales with win rate and reliability
        base = 20000 + 80000 * wr * conf
        
        # Performance boost
        perf_impact = f["ev"] * p["ev_alpha"]
        amount = base + 120000 * max(0, perf_impact)
        
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        streamer = event.streamer
        p = self.profiles[streamer]
        
        winner = next((o for o in event.outcomes if o.outcome_id == event.winner_outcome_id), None)
        if not winner: return

        wf = build_features(winner, state.prior_win_rate(streamer, winner.odds), event_outcomes=event.outcomes)
        
        # Update Alphas
        # Use simple gradient descent on the signals that would have picked the winner
        p["ev_alpha"] += self.lr * (wf["ev"] - p["ev_alpha"])
        p["whale_alpha"] += self.lr * ((winner.points_pct - winner.users_pct) - p["whale_alpha"])
        p["crowd_alpha"] += self.lr * (wf["efficiency"] - p["crowd_alpha"])
        
        p["wins"] += 1 if won else 0
        p["total"] += 1
        p["n_events"] += 1
        p["reliability"] = min(1.0, p["n_events"] / 10.0)
