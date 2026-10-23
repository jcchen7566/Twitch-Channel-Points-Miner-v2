"""v2 variant — centroid-sim gate, NO learned temperature/biases.

Tests whether the learned gate parameters add noise or value.
Gate = centroid cosine similarities normalized to probabilities.
"""

import math

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from ..core.strategy_base import StrategyBase
from ..core.features import build_features, clamp


def _keyword_cluster(title: str) -> int:
    t = (title or "").strip().lower()
    if any(k in t for k in ["誰會贏", "哪邊會贏", " vs ", "誰贏"]):
        return 0
    if any(k in t for k in ["失分", "勝投", "全壘打", "安打", "觸身", "er ", "er)"]):
        return 1
    if any(k in t for k in ["命一關不刷", "一關不刷"]):
        return 2
    if any(k in t for k in ["第幾名", "開到幾星", "抽"]):
        return 3
    if any(k in t for k in ["會不會過關", "能不能過關", "一把決勝負"]):
        return 4
    if "最穩的那台" in t:
        return 5
    if "丟骨頭" in t:
        return 6
    return 7


CLUSTER_NAMES = ["WHO_WIN", "BASEBALL", "LIFE_GAMES", "NUMBER_GUESS", "CHALLENGE", "LADDER", "BONE", "CATCH_ALL"]
N_CLUSTERS = len(CLUSTER_NAMES)


class Slrabbit99ClusterEnsembleV2Nogate(StrategyBase):
    name = "slrabbit99_cluster_ensemble_v2_nogate"

    FEATURE_NAMES = [
        "bias", "log_odds", "users_pct", "points_pct", "ratio",
        "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
        "p", "ev", "conf", "momentum", "crowd_liq",
        "users_minus_points", "points_minus_users", "efficiency",
    ]

    def __init__(self):
        super().__init__()
        self.cluster_weights = {}
        self._fitted = False

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

    def preload_history(self, events):
        titles = [e.main_title for e in events if e.streamer == "slrabbit99" and e.main_title]
        titles = list(dict.fromkeys(titles))
        if not titles:
            return
        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=2000, binary=True)
        n_dims = min(20, len(titles) - 1)
        self.svd = TruncatedSVD(n_components=max(1, n_dims), random_state=42)
        X = self.vectorizer.fit_transform(titles)
        Xr = self.svd.fit_transform(X)
        centroids = {}
        for i, t in enumerate(titles):
            c = _keyword_cluster(t)
            centroids.setdefault(c, []).append(Xr[i])
        self.centroids_ = {}
        for c in range(N_CLUSTERS):
            v = centroids.get(c, [])
            self.centroids_[c] = np.mean(v, axis=0) if v else np.zeros(max(1, n_dims))
        self._fitted = True

    def _gate_weights(self, title: str) -> list[float]:
        if not self._fitted or not title:
            return [1.0 / N_CLUSTERS] * N_CLUSTERS
        X = self.vectorizer.transform([title])
        Xr = self.svd.transform(X)
        vec = Xr[0]
        sims = []
        for c in range(N_CLUSTERS):
            d = float(np.dot(vec, self.centroids_[c]))
            n = float(np.linalg.norm(vec) * np.linalg.norm(self.centroids_[c])) + 1e-10
            sims.append(d / n)
        peak = max(sims)
        exps = [math.exp(max(-30.0, min(30.0, 6.0 * (s - peak)))) for s in sims]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

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
        return sum(w * feat[n] for w, n in zip(weights, self.FEATURE_NAMES))

    def _ensemble_scores(self, event, state):
        gate = self._gate_weights(event.main_title)
        feats = [self._features(event, o, state) for o in event.outcomes]
        scores = [0.0] * len(event.outcomes)
        for c in range(N_CLUSTERS):
            for j in range(len(event.outcomes)):
                scores[j] += gate[c] * self._score_outcome(self.cluster_weights[c], feats[j])
        return feats, scores, self._softmax(scores), gate

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        _, _, probs, _ = self._ensemble_scores(event, state)
        best, best_edge = None, -10**9
        for o, p in zip(event.outcomes, probs):
            e = p * o.odds - 1.0
            if e > best_edge:
                best_edge, best = e, o
        return best

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99" or selected is None:
            return 0.0
        _, scores, probs, _ = self._ensemble_scores(event, state)
        idx = next((i for i, o in enumerate(event.outcomes) if o.outcome_id == selected.outcome_id), None)
        if idx is None:
            return 0.0
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        edge = probs[idx] * selected.odds - 1.0
        sig = 1.0 / (1.0 + math.exp(-margin)) if margin < 35 else 1.0
        return clamp(self.base_bet + self.edge_mult * max(0.0, edge) * sig, 20000.0, 120000.0)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return
        cidx = _keyword_cluster(event.main_title)
        w = self.cluster_weights[cidx]
        feats, scores, probs, _ = self._ensemble_scores(event, state)
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]
        for i, name in enumerate(self.FEATURE_NAMES):
            g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
            w[i] -= self.lr * g