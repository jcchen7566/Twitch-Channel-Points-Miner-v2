import math
from collections import defaultdict
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase

class DynamicRankBias(StrategyBase):
    """
    Learns the win probability of each rank (Favorite vs Underdog) per streamer.
    Bets on the rank with the highest historical Edge (Actual WR * Odds - 1).
    """
    name = "dynamic_rank_bias"

    def on_backtest_start(self):
        # Tracking: streamer -> rank -> {wins, total}
        self.rank_stats = defaultdict(lambda: defaultdict(lambda: {"w": 0.0, "n": 0.0}))
        # Prior: assume Rank 1 (Favorite) wins 60%, Rank 2 wins 35%, Rank 3+ wins 5%
        self.priors = {1: 0.60, 2: 0.35, 3: 0.05}

    def _get_rank_prob(self, streamer, rank):
        stats = self.rank_stats[streamer][rank]
        # Bayesian smoothing with prior
        prior_n = 5.0
        prior_p = self.priors.get(rank, 0.01)
        return (stats["w"] + prior_n * prior_p) / (stats["n"] + prior_n)

    def pick(self, event, state):
        # Sort outcomes by odds to get ranks (Rank 1 = lowest odds = favorite)
        sorted_outcomes = sorted(event.outcomes, key=lambda x: x.odds)
        
        best = None
        best_edge = -10**9
        
        for i, o in enumerate(sorted_outcomes):
            rank = i + 1
            prob = self._get_rank_prob(event.streamer, rank)
            edge = prob * o.odds - 1.0
            
            if edge > best_edge:
                best_edge = edge
                best = o
        return best

    def size(self, event, selected, state):
        # Sizing based on the learned Edge
        sorted_outcomes = sorted(event.outcomes, key=lambda x: x.odds)
        rank = 1
        for i, o in enumerate(sorted_outcomes):
            if o.outcome_id == selected.outcome_id:
                rank = i + 1
                break
        
        prob = self._get_rank_prob(event.streamer, rank)
        edge = prob * selected.odds - 1.0
        
        # Fractional Kelly based on the learned rank probability
        b = max(0.01, selected.odds - 1.0)
        q = 1.0 - prob
        kelly = (prob * b - q) / b
        
        # Scale to 20k - 120k
        fractional_kelly = max(0.0, kelly) * 0.30
        amount = 20000 + 100000 * fractional_kelly
        
        # Cap high-odds risk
        if selected.odds > 6.0: amount *= 0.5
        
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        # Update rank stats for all ranks in this event
        sorted_outcomes = sorted(event.outcomes, key=lambda x: x.odds)
        for i, o in enumerate(sorted_outcomes):
            rank = i + 1
            self.rank_stats[event.streamer][rank]["n"] += 1.0
            if o.outcome_id == event.winner_outcome_id:
                self.rank_stats[event.streamer][rank]["w"] += 1.0
