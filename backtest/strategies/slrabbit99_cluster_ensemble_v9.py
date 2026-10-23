"""v9 — dynamic clustering via n-gram cosine + keyword fallback.

Replaces hardcoded keyword _cluster_idx with DynamicClusterManager that
spawns new clusters for novel titles.  Ensemble weights are proportional
to cosine similarity between title and each cluster centroid.
"""

import math

from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase
from ..core.title_clustering import DynamicClusterManager, _default_keyword_fallback

SEED_EXAMPLES = [
    ["誰會贏", "哪邊會贏", " vs ", "誰贏"],
    ["失分", "勝投", "全壘打", "安打", "觸身"],
    ["命一關不刷", "一關不刷"],
    ["第幾名", "開到幾星", "抽"],
    ["會不會過關", "能不能過關", "一把決勝負"],
    ["最穩的那台"],
    ["丟骨頭"],
]

N_SEEDS = len(SEED_EXAMPLES)

FEATURE_NAMES = [
    "bias", "log_odds", "users_pct", "points_pct", "ratio",
    "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
    "p", "ev", "conf", "momentum", "crowd_liq",
    "users_minus_points", "points_minus_users", "efficiency",
]

_INITIAL_WEIGHTS = [
    -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
    0.00, 0.00, 0.08, 0.06, 0.00, -0.04, -0.02, 0.02, 0.00,
]


def _slrabbit99_keyword_fallback(title: str) -> int | None:
    idx = _default_keyword_fallback(title)
    return idx


class Slrabbit99ClusterEnsembleV9(StrategyBase):
    name = "slrabbit99_cluster_ensemble_v9"

    def on_backtest_start(self):
        self.lr = 0.05
        self.base_bet = 20000.0
        self.edge_mult = 100000.0

        self.cluster_mgr = DynamicClusterManager(
            SEED_EXAMPLES, threshold=0.25,
            keyword_fallback=_slrabbit99_keyword_fallback,
        )
        self.cluster_weights: dict[int, list[float]] = {
            i: list(_INITIAL_WEIGHTS) for i in range(N_SEEDS)
        }

    def _get_w(self, cidx: int) -> list[float]:
        w = self.cluster_weights.get(cidx)
        if w is None:
            w = list(_INITIAL_WEIGHTS)
            self.cluster_weights[cidx] = w
        return w

    @staticmethod
    def _softmax(values):
        if not values:
            return []
        peak = max(values)
        exps = [math.exp(max(-20.0, min(20.0, v - peak))) for v in values]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

    @staticmethod
    def _sigmoid(x):
        if x < -35.0:
            return 0.0
        if x > 35.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

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
        return sum(w * feat[name] for w, name in zip(weights, FEATURE_NAMES))

    def _ensemble_scores(self, event, state):
        title = event.main_title
        cidx = self.cluster_mgr.assign(title)
        feats = [self._features(event, o, state) for o in event.outcomes]

        scores = [0.0] * len(event.outcomes)
        total_w = 0.0
        vec = self.cluster_mgr._ngrams(title)

        for c in self.cluster_weights:
            w = self._get_w(c)
            sim = max(0.01, self.cluster_mgr.clusters[c].cosine(vec))
            total_w += sim
            for j in range(len(event.outcomes)):
                scores[j] += sim * self._score_outcome(w, feats[j])

        if total_w > 0:
            scores = [s / total_w for s in scores]

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

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        cidx = self.cluster_mgr.assign(event.main_title)
        w = self._get_w(cidx)

        feats, scores, probs, _ = self._ensemble_scores(event, state)
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]

        for i, name in enumerate(FEATURE_NAMES):
            g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
            w[i] -= self.lr * g
