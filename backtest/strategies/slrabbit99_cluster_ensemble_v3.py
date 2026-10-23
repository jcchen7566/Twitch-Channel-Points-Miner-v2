"""Full-rebuild MoE for slrabbit99 — retrains from scratch on each decision.

Each decision rebuilds vectorizer → SVD → centroids → cluster weights
→ gate params from ALL historical events (self._past_events).  No
lookahead — only events resolved before the current one are used.

Runtime equivalent: query bet_logs.sqlite → rebuild → predict.
"""

import math

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from ..core.strategy_base import StrategyBase, OnlineStats
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


N_CLUSTERS = 8


class Slrabbit99ClusterEnsembleV3(StrategyBase):
    name = "slrabbit99_cluster_ensemble_v3"

    FEATURE_NAMES = [
        "bias", "log_odds", "users_pct", "points_pct", "ratio",
        "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
        "p", "ev", "conf", "momentum", "crowd_liq",
        "users_minus_points", "points_minus_users", "efficiency",
    ]

    def __init__(self):
        super().__init__()
        self.cluster_weights = {}
        self._past_events = []
        self._retrain_counter = 0

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
        pass  # v3 retrains from scratch on each pick — no preload needed

    # ── core: full rebuild from past events ──────────────────────────

    def _rebuild_title_space(self):
        """Rebuild vectorizer → SVD → centroids from all past titles."""
        events = self._past_events
        slr = [e for e in events if e.streamer == "slrabbit99" and e.main_title]
        titles = list(dict.fromkeys(e.main_title for e in slr))
        if len(titles) < 3:
            self.vectorizer = None
            return

        vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=2000, binary=True)
        n_dims = min(20, len(titles) - 1)
        svd = TruncatedSVD(n_components=max(1, n_dims), random_state=42)
        X = vec.fit_transform(titles)
        Xr = svd.fit_transform(X)

        buckets = {}
        for i, t in enumerate(titles):
            c = _keyword_cluster(t)
            buckets.setdefault(c, []).append(Xr[i])

        centroids = {}
        for c in range(N_CLUSTERS):
            v = buckets.get(c, [])
            centroids[c] = np.mean(v, axis=0) if v else np.zeros(svd.n_components)

        self.vectorizer = vec
        self.svd = svd
        self.centroids_ = centroids

    def _retrain(self):
        events = self._past_events
        slr = [e for e in events if e.streamer == "slrabbit99" and e.outcomes and e.winner_outcome_id]
        if len(slr) < 3:
            return
        if not hasattr(self, "vectorizer") or self.vectorizer is None:
            return

        w = {c: list(self.cluster_weights[c]) for c in range(N_CLUSTERS)}
        slr_by_cluster = {c: [] for c in range(N_CLUSTERS)}
        for e in slr:
            slr_by_cluster[_keyword_cluster(e.main_title)].append(e)

        for _ in range(2):
            rp = OnlineStats()
            for c in range(N_CLUSTERS):
                for e in slr_by_cluster[c]:
                    feats = [self._features(e, o, rp) for o in e.outcomes]
                    scores = [self._score_outcome(w[c], f) for f in feats]
                    probs = self._softmax(scores)
                    y = [1.0 if o.outcome_id == e.winner_outcome_id else 0.0 for o in e.outcomes]
                    grad = [p - yy for p, yy in zip(probs, y)]
                    for i, name in enumerate(self.FEATURE_NAMES):
                        g = sum(grad[j] * feats[j][name] for j in range(len(e.outcomes)))
                        w[c][i] -= 0.01 * g
                    winner_odds = next(
                        (o.odds for o in e.outcomes if o.outcome_id == e.winner_outcome_id), 2.0
                    )
                    rp.update(e.streamer, winner_odds, True)

        self.cluster_weights = w

    # ── gate from scratch (no self._fitted dependency) ───────────────

    @staticmethod
    def _gate_from_scratch(title, log_temp, logits, centroids, vec, svd):
        if not title:
            return [1.0 / N_CLUSTERS] * N_CLUSTERS
        try:
            X = vec.transform([title])
            Xr = svd.transform(X)
        except Exception:
            return [1.0 / N_CLUSTERS] * N_CLUSTERS
        vec_t = Xr[0]
        tau = math.exp(log_temp)
        vals = []
        for c in range(N_CLUSTERS):
            center = centroids.get(c)
            if center is None:
                vals.append(logits[c])
                continue
            d = float(np.dot(vec_t, center))
            n1 = float(np.linalg.norm(vec_t))
            n2 = float(np.linalg.norm(center))
            sim = d / (n1 * n2 + 1e-10)
            vals.append(tau * sim + logits[c])
        peak = max(vals) if vals else 0.0
        exps = [math.exp(max(-30.0, min(30.0, v - peak))) for v in vals]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

    # ── helpers (same as v2) ─────────────────────────────────────────

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

    @staticmethod
    def _score_outcome(weights, feat):
        F = ["bias", "log_odds", "users_pct", "points_pct", "ratio",
             "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
             "p", "ev", "conf", "momentum", "crowd_liq",
             "users_minus_points", "points_minus_users", "efficiency"]
        return sum(w * feat[n] for w, n in zip(weights, F))

    def _ensemble_scores(self, event, state):
        gate = self._gate_weights(event.main_title)
        feats = [self._features(event, o, state) for o in event.outcomes]
        scores = [0.0] * len(event.outcomes)
        for c in range(N_CLUSTERS):
            w = self.cluster_weights[c]
            for j in range(len(event.outcomes)):
                scores[j] += gate[c] * self._score_outcome(w, feats[j])
        probs = self._softmax(scores)
        return feats, scores, probs, gate

    def _gate_weights(self, title: str) -> list[float]:
        if not hasattr(self, "vectorizer") or self.vectorizer is None or not title:
            return [1.0 / N_CLUSTERS] * N_CLUSTERS
        return self._gate_from_scratch(
            title, self.gate_log_temp, self.cluster_logits,
            self.centroids_, self.vectorizer, self.svd,
        )

    # ── inference ────────────────────────────────────────────────────

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        self._retrain_counter += 1
        if self._retrain_counter % 10 == 0:
            self._retrain()
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
        sig = 1.0 / (1.0 + math.exp(-margin)) if margin < 35 else 1.0
        amount = self.base_bet + self.edge_mult * max(0.0, edge) * sig
        return clamp(amount, 20000.0, 120000.0)

    # ── event accumulation ──────────────────────────────────────────

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return
        self._past_events.append(event)