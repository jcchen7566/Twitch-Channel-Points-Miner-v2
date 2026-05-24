import math
from collections import defaultdict
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase
from ..core.title_clustering import cluster_title

class MultiAlphaRegimeExpert(StrategyBase):
    """
    Advanced Multi-Expert Strategy:
    1. Clusters by Game Regime (LOL, TFT, etc).
    2. Maintains a 'Committee of Alphas' per [Streamer x Regime].
    3. Uses an Online Softmax Bandit to weight experts.
    """
    name = "multi_alpha_regime_expert"

    def on_backtest_start(self):
        # Key: (streamer, regime) -> expert_weights
        self.expert_stats = defaultdict(lambda: {
            "weights": [1.0, 0.2, 0.1, 0.3], # Start with strong EV trust
            "n_events": 0
        })
        self.lr = 0.1

    def _get_expert_scores(self, event, outcome, state):
        prior = state.prior_win_rate(event.streamer, outcome.odds)
        f = build_features(outcome, prior, event_outcomes=event.outcomes)
        
        # Expert 1: Bayesian Shrinkage EV (The anchor)
        n = max(0.0, outcome.ht)
        lam = n / (n + 15.0)
        p_shrink = lam * f["p"] + (1.0 - lam) * prior
        ev_shrink = p_shrink * outcome.odds - 1.0
        
        # Expert 2: Whale Alpha (Points concentration)
        whale_alpha = (outcome.points_pct - outcome.users_pct) / 100.0
        
        # Expert 3: Underdog Alpha (Rank 2)
        sorted_o = sorted(event.outcomes, key=lambda x: x.odds)
        is_rank2 = 1.0 if (len(sorted_o) > 1 and outcome.outcome_id == sorted_o[1].outcome_id) else 0.0
        
        return [ev_shrink, whale_alpha, is_rank2]

    def pick(self, event, state):
        regime = cluster_title(event.main_title)
        stats = self.expert_stats[(event.streamer, regime)]
        weights = stats["weights"]
        
        best = None
        best_score = -10**9
        
        for o in event.outcomes:
            scores = self._get_expert_scores(event, o, state)
            # score = EV_Safe + W2 * Whale + W3 * Underdog
            score = scores[0] + weights[1] * scores[1] + weights[2] * scores[2]
            
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        regime = cluster_title(event.main_title)
        stats = self.expert_stats[(event.streamer, regime)]
        
        scores = self._get_expert_scores(event, selected, state)
        weights = stats["weights"]
        
        # Final estimated Edge
        edge = scores[0] + weights[1] * scores[1] + weights[2] * scores[2]
        
        # Sizing: Half-Kelly style (Approx)
        # Scale 40k to 120k based on edge
        amount = 45000 + 150000 * max(0, edge)
        
        # Confidence multiplier
        n = stats["n_events"]
        conf = min(1.0, 0.5 + n / 10.0)
        
        return clamp(amount * conf, 20000, 120000)

    def on_event_end(self, event, selected, won, pnl, state):
        regime = cluster_title(event.main_title)
        stats = self.expert_stats[(event.streamer, regime)]
        
        winner_id = event.winner_outcome_id
        winner = next((o for o in event.outcomes if o.outcome_id == winner_id), None)
        if not winner: return
        
        winner_scores = self._get_expert_scores(event, winner, state)
        
        # Online update of Alpha weights
        # If whale_alpha was positive for the winner, increase whale trust
        stats["weights"][1] += self.lr * (winner_scores[1] - stats["weights"][1])
        # If winner was Rank 2, increase underdog trust
        stats["weights"][2] += self.lr * (winner_scores[2] - stats["weights"][2])
        
        stats["n_events"] += 1
