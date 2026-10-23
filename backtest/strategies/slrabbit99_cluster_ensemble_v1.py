"""Title clustering mixture-of-experts for slrabbit99.

Keyword-based cluster assignment, each cluster has its own 17-weight
softmax vector.  Hard assignment during training (each event belongs
to exactly one cluster), softmax gating during inference.
"""

import math

from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase


def _cluster_idx(title: str) -> int:
    t = (title or "").strip().lower()
    # Order matters — first match wins
    if any(k in t for k in ["誰會贏", "哪邊會贏", " vs ", "誰贏"]):
        return 0  # WHO_WIN
    if any(k in t for k in ["失分", "勝投", "全壘打", "安打", "觸身", "er ", "er)"]):
        return 1  # BASEBALL_PROPS
    if any(k in t for k in ["命一關不刷", "一關不刷"]):
        return 2  # LIFE_GAMES
    if any(k in t for k in ["第幾名", "開到幾星", "抽"]):
        return 3  # NUMBER_GUESS
    if any(k in t for k in ["會不會過關", "能不能過關", "一把決勝負"]):
        return 4  # CHALLENGE
    if "最穩的那台" in t:
        return 5  # LADDER — "過X關 最多刷N次"
    if "丟骨頭" in t:
        return 6  # BONE — 骨頭挑戰
    return 7  # CATCH_ALL


CLUSTER_NAMES = [
    "WHO_WIN", "BASEBALL", "LIFE_GAMES",
    "NUMBER_GUESS", "CHALLENGE", "LADDER",
    "BONE", "CATCH_ALL",
]
N_CLUSTERS = len(CLUSTER_NAMES)


class Slrabbit99ClusterEnsembleV1(StrategyBase):
    name = "slrabbit99_cluster_ensemble_v1"

    FEATURE_NAMES = [
        "bias", "log_odds", "users_pct", "points_pct", "ratio",
        "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
        "p", "ev", "conf", "momentum", "crowd_liq",
        "users_minus_points", "points_minus_users", "efficiency",
    ]

    def __init__(self):
        super().__init__()
        self.cluster_weights = {}

    def on_backtest_start(self):
        self.lr = 0.05
        self.base_bet = 20000.0
        self.edge_mult = 100000.0
        self._initial_weights = [
            -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
            0.00, 0.00, 0.08, 0.06, 0.00, -0.04, -0.02, 0.02, 0.00,
        ]
        self.cluster_weights = {
            c: list(self._initial_weights) for c in range(N_CLUSTERS)
        }

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _softmax(values):
        if not values:
            return []
        peak = max(values)
        exps = [math.exp(max(-20.0, min(20.0, v - peak))) for v in values]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

    def _features(self, event, outcome, state):
        f = build_features(
            outcome, state.prior_win_rate(event.streamer, outcome.odds),
            event_outcomes=event.outcomes,
        )
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

    def _score_outcome(self, weights, feat):
        return sum(w * feat[name] for w, name in zip(weights, self.FEATURE_NAMES))

    # ── inference (ensemble: all clusters vote, weighted by similarity) ─

    def _ensemble_scores(self, event, state):
        cidx = _cluster_idx(event.main_title)
        feats = [self._features(event, o, state) for o in event.outcomes]

        # Ensemble: weighted by cluster similarity to event
        scores = [0.0] * len(event.outcomes)
        total_w = 0.0
        for c in range(N_CLUSTERS):
            w = self.cluster_weights[c]
            sim = 0.6 if c == cidx else 0.4 / (N_CLUSTERS - 1 + 1e-10)
            total_w += sim
            for j in range(len(event.outcomes)):
                scores[j] += sim * self._score_outcome(w, feats[j])

        probs = self._softmax(scores)
        return feats, scores, probs, cidx

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        _, _, probs, _ = self._ensemble_scores(event, state)
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
        _, scores, probs, _ = self._ensemble_scores(event, state)
        sel_idx = next(
            (i for i, o in enumerate(event.outcomes) if o.outcome_id == selected.outcome_id), None,
        )
        if sel_idx is None:
            return 0.0
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        edge = probs[sel_idx] * selected.odds - 1.0
        amount = self.base_bet + self.edge_mult * max(0.0, edge) * self._sigmoid(margin)
        return clamp(amount, 20000.0, 120000.0)

    @staticmethod
    def _sigmoid(x):
        if x < -35.0:
            return 0.0
        if x > 35.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    # ── online gradient update (only the assigned cluster) ────────────

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        cidx = _cluster_idx(event.main_title)
        w = self.cluster_weights[cidx]

        feats, scores, probs, _ = self._ensemble_scores(event, state)
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]

        for i, name in enumerate(self.FEATURE_NAMES):
            g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
            w[i] -= self.lr * g