"""Soft-gated MoE for slrabbit99 — keyword-seeded centroids + learned gate.

Training: hard assignment (only primary cluster updated, like v1).
Inference: learned softmax gate over centroid similarities + per-cluster bias.
New/unseen titles get soft assignments via centroid cosine similarity.
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


CLUSTER_NAMES = [
    "WHO_WIN", "BASEBALL", "LIFE_GAMES",
    "NUMBER_GUESS", "CHALLENGE", "LADDER",
    "BONE", "CATCH_ALL",
]
N_CLUSTERS = len(CLUSTER_NAMES)
_CLUSTER_L2 = [1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]


class Slrabbit99ClusterEnsembleV2Bins(StrategyBase):
    name = "slrabbit99_cluster_ensemble_v2_bins"

    FEATURE_NAMES = [
        "bias", "log_odds", "users_pct", "points_pct", "ratio",
        "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
        "p", "ev",
        "crowd_liq",
        "users_minus_points", "points_minus_users", "n_out",
        "liq_low", "liq_mid", "odds_high", "whale",
        "streak",
    ]

    def __init__(self):
        super().__init__()
        self.cluster_weights = {}
        self._fitted = False
        self._pnl_window = []
        self._streak = 0

    def on_backtest_start(self):
        self.lr = 0.05
        self.base_bet = 20000.0
        self.edge_mult = 100000.0
        self.gate_log_temp = math.log(8.0)
        self.cluster_logits = [0.0] * N_CLUSTERS
        self._initial_weights = [
            -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
            0.00, 0.00, 0.08, -0.04, -0.02, 0.02,
            0.00, 0.00, 0.00, 0.00, 0.00,
            0.00,
        ]
        self.cluster_weights = {
            c: list(self._initial_weights) for c in range(N_CLUSTERS)
        }

    def preload_history(self, events):
        titles = [e.main_title for e in events if e.streamer == "slrabbit99" and e.main_title]
        titles = list(dict.fromkeys(titles))
        if not titles:
            return

        self.vectorizer = CountVectorizer(
            analyzer="char_wb", ngram_range=(2, 3),
            max_features=2000, binary=True,
        )
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
            vecs = centroids.get(c, [])
            self.centroids_[c] = np.mean(vecs, axis=0) if vecs else np.zeros(max(1, n_dims))

        self._fitted = True

    # ── gating ──────────────────────────────────────────────────────

    def _gate_weights(self, title: str) -> list[float]:
        if not self._fitted or not title:
            return [1.0 / N_CLUSTERS] * N_CLUSTERS

        X = self.vectorizer.transform([title])
        Xr = self.svd.transform(X)
        vec = Xr[0]

        tau = math.exp(self.gate_log_temp)
        logits = []
        for c in range(N_CLUSTERS):
            center = self.centroids_[c]
            d = float(np.dot(vec, center))
            n1 = float(np.linalg.norm(vec))
            n2 = float(np.linalg.norm(center))
            sim = d / (n1 * n2 + 1e-10)
            logits.append(tau * sim + self.cluster_logits[c])

        peak = max(logits) if logits else 0.0
        exps = [math.exp(max(-30.0, min(30.0, v - peak))) for v in logits]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

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
        tp = outcome.total_points
        up = outcome.users_pct
        pp = outcome.points_pct
        return {
            "bias": 1.0,
            "log_odds": f["log_odds"],
            "users_pct": up / 100.0,
            "points_pct": pp / 100.0,
            "ratio": f["ratio"],
            "rel_ratio": f["rel_ratio"],
            "rel_odds": f["rel_odds"],
            "rel_surprise": f["rel_surprise"],
            "log_liq": f["log_liq"],
            "p": f["p"],
            "ev": f["ev"],
            "crowd_liq": f["crowd_liq"] / 10.0,
            "users_minus_points": (up - pp) / 100.0,
            "points_minus_users": (pp - up) / 100.0,
            "n_out": 1.0 / max(len(event.outcomes), 2),
            "liq_low": 1.0 if tp > 0 and tp < 500000 else 0.0,
            "liq_mid": 1.0 if 500000 <= tp <= 2000000 else 0.0,
            "odds_high": 1.0 if outcome.odds > 1.8 else 0.0,
            "whale": 1.0 if pp > 0 and (up / pp) < 0.9 else 0.0,
            "streak": self._streak / 10.0,
        }

    @staticmethod
    def _score_outcome(weights, feat):
        F = ["bias", "log_odds", "users_pct", "points_pct", "ratio",
             "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
             "p", "ev", "crowd_liq",
              "users_minus_points", "points_minus_users", "n_out",
              "liq_low", "liq_mid", "odds_high", "whale",
              "streak"]
        return sum(w * feat[n] for w, n in zip(weights, F))

    def _ensemble_scores(self, event, state, best_only=False):
        gate = self._gate_weights(event.main_title)
        feats = [self._features(event, o, state) for o in event.outcomes]
        scores = [0.0] * len(event.outcomes)

        if best_only:
            best_c = max(range(N_CLUSTERS), key=lambda c: gate[c])
            w = self.cluster_weights[best_c]
            for j in range(len(event.outcomes)):
                scores[j] = self._score_outcome(w, feats[j])
        else:
            for c in range(N_CLUSTERS):
                w = self.cluster_weights[c]
                for j in range(len(event.outcomes)):
                    scores[j] += gate[c] * self._score_outcome(w, feats[j])

        probs = self._softmax(scores)
        return feats, scores, probs, gate

    # ── inference ────────────────────────────────────────────────────

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        _, _, probs, _ = self._ensemble_scores(event, state, best_only=True)
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
        _, scores, probs, _ = self._ensemble_scores(event, state, best_only=True)
        sel_idx = next(
            (i for i, o in enumerate(event.outcomes) if o.outcome_id == selected.outcome_id), None,
        )
        if sel_idx is None:
            return 0.0
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        edge = probs[sel_idx] * selected.odds - 1.0
        sig = 1.0 / (1.0 + math.exp(-margin)) if margin < 35 else 1.0
        amount = self.base_bet + self.edge_mult * max(0.0, edge) * sig

        # PnL modulation: compound wins, cut losses
        n_pnl = len(self._pnl_window)
        if n_pnl >= 5:
            total_pnl = sum(self._pnl_window)
            total_bet = sum(abs(x) for x in self._pnl_window)  # approximate
            recent_roi = total_pnl / (total_bet + 1e-6)
            mod = 1.0 + 0.4 * max(-0.8, min(0.8, recent_roi))
            amount *= mod

        return clamp(amount, 20000.0, 120000.0)

    # ── online training ──────────────────────────────────────────────

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        self._pnl_window.append(pnl)
        if len(self._pnl_window) > 30:
            self._pnl_window.pop(0)

        cidx = _keyword_cluster(event.main_title)
        w = self.cluster_weights[cidx]

        # Streak tracking
        if pnl > 0:
            self._streak = min(10, self._streak + 1)
        else:
            self._streak = max(-10, self._streak - 1)

        feats, scores, probs, gate = self._ensemble_scores(event, state)
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]

        for i, name in enumerate(self.FEATURE_NAMES):
            g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
            l2_c = _CLUSTER_L2[cidx]
            w[i] -= self.lr * (g + l2_c * w[i])

        # Gate: learn temperature + per-cluster logits
        margin_loss = sum(abs(g) for g in grad) / len(grad) if grad else 0.0
        self.gate_log_temp += 0.01 * (1.0 - 2.0 * margin_loss)
        self.gate_log_temp = max(-3.0, min(10.0, self.gate_log_temp))

        for c in range(N_CLUSTERS):
            c_score = sum(
                self._score_outcome(self.cluster_weights[c], feats[j]) * grad[j]
                for j in range(len(event.outcomes))
            )
            self.cluster_logits[c] += 0.002 * c_score
            self.cluster_logits[c] = max(-3.0, min(3.0, self.cluster_logits[c]))