import math

from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase


class Slrabbit99FeatureSoftmaxV1(StrategyBase):
    """
    slrabbit99-specific online multinomial model.

    This strategy uses only log-derived market / crowd features and learns a
    per-streamer softmax scorer over each event's outcomes.
    """

    name = "slrabbit99_feature_softmax_v1"

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
        # Fixed from the data sweep:
        # - online softmax update
        # - per-streamer weights
        # - light skip threshold on weak edges
        self.weights = {}
        self.lr = 0.035
        self.reg = 1e-4
        self.min_edge = 0.03
        self.base_bet = 20000.0
        self.edge_mult = 100000.0
        self.odds_penalty_cut = 5.0
        self.ratio_penalty_cut = 1.0
        self.conf_penalty_cut = 0.5
        self._initial_weights = [
            -0.10,  # bias
            0.10,   # log_odds
            0.00,   # users_pct
            0.00,   # points_pct
            -0.04,  # ratio
            -0.10,  # rel_ratio
            0.00,   # rel_odds
            -0.06,  # rel_surprise
            0.00,   # log_liq
            0.00,   # p
            0.08,   # ev
            0.06,   # conf
            0.00,   # momentum
            -0.04,  # crowd_liq
            -0.02,  # users_minus_points
            0.02,   # points_minus_users
            0.00,   # efficiency
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

        if best is None or best_edge < self.min_edge:
            return None
        return best

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99" or selected is None:
            return 0.0

        _, features, _, probs = self._event_prediction(event, state)
        selected_index = next(
            (i for i, outcome in enumerate(event.outcomes) if outcome.outcome_id == selected.outcome_id),
            None,
        )
        if selected_index is None:
            return 0.0

        feat = features[selected_index]
        edge_proxy = max(0.0, probs[selected_index] * selected.odds - 1.0)
        amount = self.base_bet + self.edge_mult * edge_proxy

        if selected.odds > self.odds_penalty_cut:
            amount *= 0.8
        if feat["ratio"] > self.ratio_penalty_cut:
            amount *= 0.85
        if feat["conf"] < self.conf_penalty_cut:
            amount *= 0.85

        return clamp(amount, 20000.0, 120000.0)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        weights, feats, _, probs = self._event_prediction(event, state)

        # Multinomial softmax gradient update against the actual winner.
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]
        for i, name in enumerate(self.FEATURE_NAMES):
            g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
            weights[i] -= self.lr * (g + self.reg * weights[i])
