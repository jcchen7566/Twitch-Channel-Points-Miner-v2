import math
from collections import defaultdict
from ..core.features import build_features, odds_regime
from ..core.strategy_base import StrategyBase


class StreamerSpecialist(StrategyBase):
    """
    A strategy that learns per-streamer feature importance.
    Falls back to global priors when data is sparse.
    """
    name = "streamer_specialist"

    def on_backtest_start(self):
        # Per-streamer weights: [bias, d_log_odds, d_ratio, d_ev, conf, users_minus_points]
        self.streamer_weights = defaultdict(lambda: [0.0] * 6)
        self.streamer_counts = defaultdict(int)
        
        # Global weights as fallback (pre-tuned from global analysis)
        self.global_weights = [-0.26, -0.72, -0.10, 0.03, 0.17, -0.03]
        
        self.lr = 0.15 # Increased learning rate for faster adaptation
        self.reg = 1e-4 # Lowered regularization

    def _get_weights(self, streamer):
        count = self.streamer_counts[streamer]
        s_w = self.streamer_weights[streamer]
        
        # More aggressive fallback: use streamer weights much earlier
        lam = min(1.0, max(0.0, (count - 2) / 8.0))
        return [lam * sw + (1.0 - lam) * gw for sw, gw in zip(s_w, self.global_weights)]

    def _feat_vec(self, event, outcome, state):
        # Calculate mean features for the event for delta calculation
        outcomes_feat = [build_features(o, state.prior_win_rate(event.streamer, o.odds)) for o in event.outcomes]
        mean_log_odds = sum(math.log(max(1.01, o.odds)) for o in event.outcomes) / len(event.outcomes)
        mean_ratio = sum(f["ratio"] for f in outcomes_feat) / len(outcomes_feat)
        mean_ev = sum(f["ev"] for f in outcomes_feat) / len(outcomes_feat)
        
        f = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds))
        return [
            1.0,
            math.log(max(1.01, outcome.odds)) - mean_log_odds,
            f["ratio"] - mean_ratio,
            f["ev"] - mean_ev,
            f["conf"],
            outcome.users_pct - outcome.points_pct
        ]

    def _predict(self, streamer, x):
        w = self._get_weights(streamer)
        z = sum(wi * xi for wi, xi in zip(w, x))
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def pick(self, event, state):
        best = None
        best_score = -10**9
        for o in event.outcomes:
            x = self._feat_vec(event, o, state)
            p = max(0.01, min(0.99, self._predict(event.streamer, x)))
            # Model-driven selection
            score = p * o.odds - 1.0
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        x = self._feat_vec(event, selected, state)
        p = max(0.01, min(0.99, self._predict(event.streamer, x)))
        ev = p * selected.odds - 1.0
        
        # Aggressive sizing to capitalize on learned patterns
        count = self.streamer_counts[event.streamer]
        base_size = 80000 if count > 5 else 40000
        multiplier = 150000 if count > 5 else 90000
        
        amount = base_size + multiplier * ev
        if p < 0.40: amount *= 0.8
        if selected.odds > 5.0: amount *= 0.75
        
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        streamer = event.streamer
        x = self._feat_vec(event, selected, state)
        p = self._predict(streamer, x)
        y = 1.0 if won else 0.0
        g = p - y
        
        # Online update for the streamer specific weights
        w = self.streamer_weights[streamer]
        for i in range(len(w)):
            w[i] -= self.lr * (g * x[i] + self.reg * w[i])
            
        self.streamer_counts[streamer] += 1
