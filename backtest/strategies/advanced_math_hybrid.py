import math
from collections import defaultdict
from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase

class AdvancedMathHybrid(StrategyBase):
    """
    Hybrid strategy combining:
    - Per-streamer learned weights (online logistic regression)
    - High-order feature interactions
    - Fractional Kelly sizing with safety buffers
    """
    name = "advanced_math_hybrid"

    def on_backtest_start(self):
        # Weights: [bias, rel_odds, rel_ratio, efficiency, conf, crowd_liq]
        self.weights = defaultdict(lambda: [0.0, -1.0, -0.5, 0.2, 0.1, -0.1])
        self.counts = defaultdict(int)
        self.lr = 0.12
        self.reg = 1e-4

    def _get_x(self, event, outcome, state):
        f = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        return [
            1.0,
            f["rel_odds"],
            f["rel_ratio"],
            f["efficiency"],
            f["conf"],
            f["crowd_liq"]
        ]

    def _predict(self, streamer, x):
        w = self.weights[streamer]
        z = sum(wi * xi for wi, xi in zip(w, x))
        z = clamp(z, -20, 20)
        return 1.0 / (1.0 + math.exp(-z))

    def pick(self, event, state):
        best = None
        best_ev = -10**9
        for o in event.outcomes:
            x = self._get_x(event, o, state)
            p = max(0.01, min(0.99, self._predict(event.streamer, x)))
            ev = p * o.odds - 1.0
            if ev > best_ev:
                best_ev = ev
                best = o
        return best

    def size(self, event, selected, state):
        x = self._get_x(event, selected, state)
        p = max(0.01, min(0.99, self._predict(event.streamer, x)))
        
        # Sizing: Fractional Kelly with confidence scaling
        # k = (p * b - q) / b where b = odds - 1
        b = max(0.01, selected.odds - 1.0)
        q = 1.0 - p
        kelly = (p * b - q) / b
        
        # 0.25 Fractional Kelly
        fractional_kelly = max(0.0, kelly) * 0.25
        
        # Scaled to points range 20k - 120k
        f = build_features(selected, state.prior_win_rate(event.streamer, selected.odds), event_outcomes=event.outcomes)
        base = 20000
        extra = 100000 * fractional_kelly * f["conf"]
        
        # Meta-Risk: odds regime penalty
        if selected.odds > 5.0: extra *= 0.8
        
        return base + extra

    def on_event_end(self, event, selected, won, pnl, state):
        streamer = event.streamer
        x = self._get_x(event, selected, state)
        p = self._predict(streamer, x)
        y = 1.0 if won else 0.0
        
        # Gradient update
        w = self.weights[streamer]
        grad = p - y
        for i in range(len(w)):
            w[i] -= self.lr * (grad * x[i] + self.reg * w[i])
        
        self.counts[streamer] += 1
