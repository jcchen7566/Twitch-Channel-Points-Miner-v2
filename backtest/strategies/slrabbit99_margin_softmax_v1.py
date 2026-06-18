import math

from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase


class Slrabbit99MarginSoftmaxV1(StrategyBase):
    """
    slrabbit99-specific online softmax model with margin-calibrated sizing.

    Compared to the plain feature softmax, the selection logic stays the same,
    but bet size is scaled by the score margin between the top and runner-up
    outcome. This is a light-weight calibration layer inspired by ranking
    calibration / confidence weighting methods.
    """

    name = "slrabbit99_margin_softmax_v1"

    FEATURE_NAMES = [
        "bias",
        "log_odds",
        "users_pct",
        "points_pct",
        "ratio",
        "rel_ratio",
        "rel_odds",
        "rel_surprise",
        "log_liq",
        "p",
        "ev",
        "conf",
        "momentum",
        "crowd_liq",
        "users_minus_points",
        "points_minus_users",
        "efficiency",
    ]

    def on_backtest_start(self):
        self.weights = {}
        self.lr = 0.035
        self.reg = 1e-4
        self.base_bet = 20000.0
        self.edge_mult = 100000.0
        self._initial_weights = [
            -0.10,
            0.10,
            0.00,
            0.00,
            -0.04,
            -0.10,
            0.00,
            -0.06,
            0.00,
            0.00,
            0.08,
            0.06,
            0.00,
            -0.04,
            -0.02,
            0.02,
            0.00,
        ]

    def _get_weights(self, streamer: str):
        if streamer not in self.weights:
            self.weights[streamer] = list(self._initial_weights)
        return self.weights[streamer]

    def _softmax(self, values: list[float]) -> list[float]:
        if not values:
            return []
        peak = max(values)
        exps = [math.exp(max(-20.0, min(20.0, v - peak))) for v in values]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

    def _features(self, event, outcome, state) -> dict[str, float]:
        f = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        return {
            "bias": 1.0,
            "log_odds": f["log_odds"],
            "users_pct": outcome.users_pct / 100.0,
            "points_pct": outcome.points_pct / 100.0,
            "ratio": f["ratio"],
            "rel_ratio": f["rel_ratio"],
            "rel_odds": f["rel_odds"],
            "rel_surprise": f["rel_surprise"],
            "log_liq": f["log_liq"],
            "p": f["p"],
            "ev": f["ev"],
            "conf": f["conf"],
            "momentum": f["momentum"],
            "crowd_liq": f["crowd_liq"] / 10.0,
            "users_minus_points": (outcome.users_pct - outcome.points_pct) / 100.0,
            "points_minus_users": (outcome.points_pct - outcome.users_pct) / 100.0,
            "efficiency": f["efficiency"],
        }

    def _score_outcome(self, weights: list[float], feat: dict[str, float]) -> float:
        return sum(w * feat[name] for w, name in zip(weights, self.FEATURE_NAMES))

    def _event_prediction(self, event, state):
        weights = self._get_weights(event.streamer)
        features = [self._features(event, o, state) for o in event.outcomes]
        scores = [self._score_outcome(weights, f) for f in features]
        probs = self._softmax(scores)
        return weights, features, scores, probs

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None

        _, _, _, probs = self._event_prediction(event, state)
        best = None
        best_edge = -10**9
        for outcome, prob in zip(event.outcomes, probs):
            edge = prob * outcome.odds - 1.0
            if edge > best_edge:
                best_edge = edge
                best = outcome
        return best

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99" or selected is None:
            return 0.0

        _, _, scores, probs = self._event_prediction(event, state)
        selected_index = next(
            (i for i, outcome in enumerate(event.outcomes) if outcome.outcome_id == selected.outcome_id),
            None,
        )
        if selected_index is None:
            return 0.0

        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        edge = probs[selected_index] * selected.odds - 1.0
        amount = self.base_bet + self.edge_mult * max(0.0, edge) * self._sigmoid(margin)
        return clamp(amount, 20000.0, 120000.0)

    def _sigmoid(self, x: float) -> float:
        if x < -35.0:
            return 0.0
        if x > 35.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        weights, feats, _, probs = self._event_prediction(event, state)
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]
        for i, name in enumerate(self.FEATURE_NAMES):
            g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
            weights[i] -= self.lr * (g + self.reg * weights[i])
