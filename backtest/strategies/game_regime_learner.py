import math
from collections import defaultdict
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase
from ..core.title_clustering import cluster_title

class GameRegimeLearner(StrategyBase):
    """
    Learns the optimal strategy parameter (Bias towards Rank 2)
    per [Streamer x Game Regime].
    
    This validates the insight that slrabbit99 playing LOL might have 
    different underdog-bias than slrabbit99 playing TFT.
    """
    name = "game_regime_learner"

    def on_backtest_start(self):
        # Key: (streamer, game_regime) -> stats
        self.stats = defaultdict(lambda: {
            "r1_n": 0.0, "r1_w": 0.0,
            "r2_n": 0.0, "r2_w": 0.0
        })

    def _get_rank_edge(self, streamer, regime, rank, odds):
        s = self.stats[(streamer, regime)]
        if rank == 1:
            wr = (s["r1_w"] + 3.0 * 0.6) / (s["r1_n"] + 3.0)
        else:
            wr = (s["r2_w"] + 3.0 * 0.35) / (s["r2_n"] + 3.0)
        return wr * odds - 1.0

    def pick(self, event, state):
        regime = cluster_title(event.main_title)
        sorted_o = sorted(event.outcomes, key=lambda x: x.odds)
        
        best = None
        best_edge = -10**9
        
        for i, o in enumerate(sorted_o):
            rank = i + 1
            if rank > 2: continue # Focus on top 2
            
            edge = self._get_rank_edge(event.streamer, regime, rank, o.odds)
            if edge > best_edge:
                best_edge = edge
                best = o
        return best

    def size(self, event, selected, state):
        # Scale based on learned edge for this specific game
        regime = cluster_title(event.main_title)
        sorted_o = sorted(event.outcomes, key=lambda x: x.odds)
        rank = 1
        if selected.outcome_id == sorted_o[1].outcome_id: rank = 2
        
        edge = self._get_rank_edge(event.streamer, regime, rank, selected.odds)
        amount = 40000 + 100000 * max(0, edge)
        
        # Risk control: if we have very little data for this game regime, be conservative
        n_total = self.stats[(event.streamer, regime)]["r1_n"] + self.stats[(event.streamer, regime)]["r2_n"]
        if n_total < 3:
            amount *= 0.6
            
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        regime = cluster_title(event.main_title)
        sorted_o = sorted(event.outcomes, key=lambda x: x.odds)
        
        stats = self.stats[(event.streamer, regime)]
        winner_id = event.winner_outcome_id
        
        if sorted_o[0].outcome_id == winner_id:
            stats["r1_w"] += 1.0
        elif len(sorted_o) > 1 and sorted_o[1].outcome_id == winner_id:
            stats["r2_w"] += 1.0
            
        stats["r1_n"] += 1.0
        stats["r2_n"] += 1.0
