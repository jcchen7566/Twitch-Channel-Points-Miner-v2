"""Stacking ensemble: v2 + feature_softmax + margin_softmax predictions averaged.

Each sub-strategy trains independently via on_event_end.  Prediction is the
average of their softmax probabilities.  Combines diversity without overfitting.
"""

import math

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from ..core.strategy_base import StrategyBase
from ..core.features import build_features, clamp


FEATURE_NAMES = [
    "bias", "log_odds", "users_pct", "points_pct", "ratio",
    "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
    "p", "ev", "conf", "momentum", "crowd_liq",
    "users_minus_points", "points_minus_users", "efficiency",
]
N_FEATURES = len(FEATURE_NAMES)


def _features(event, outcome, state):
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


def _score(weights, feat):
    return sum(w * feat[n] for w, n in zip(weights, FEATURE_NAMES))


def _softmax(values):
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(max(-20.0, min(20.0, v - peak))) for v in values]
    denom = sum(exps) or 1.0
    return [v / denom for v in exps]


def _sigmoid(x):
    if x < -35:
        return 0.0
    if x > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class Slrabbit99EnsembleStackV1(StrategyBase):
    name = "slrabbit99_ensemble_stack_v1"

    FEATURE_NAMES = FEATURE_NAMES

    def __init__(self):
        super().__init__()
        self._fitted = False

    def on_backtest_start(self):
        self.lr = 0.05
        self.base_bet = 20000.0
        self.edge_mult = 100000.0

        base = [-0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
                0.00, 0.00, 0.08, 0.06, 0.00, -0.04, -0.02, 0.02, 0.00]

        # Single softmax models
        self.fs_weights = list(base)       # feature_softmax
        self.ms_weights = list(base)       # margin_softmax

        # MoE cluster weights
        self.gate_log_temp = math.log(6.0)
        self.cluster_logits = [0.0] * 8
        self.cluster_weights = {c: list(base) for c in range(8)}

    # ── cluster helpers (same as v2) ──────────────────────────────────

    def _cluster_idx(self, title):
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
            c = self._cluster_idx(t)
            centroids.setdefault(c, []).append(Xr[i])
        self.centroids_ = {}
        for c in range(8):
            vecs = centroids.get(c, [])
            self.centroids_[c] = np.mean(vecs, axis=0) if vecs else np.zeros(max(1, n_dims))
        self._fitted = True

    def _gate_weights(self, title):
        if not self._fitted or not title:
            return [1.0 / 8] * 8
        X = self.vectorizer.transform([title])
        Xr = self.svd.transform(X)
        vec = Xr[0]
        tau = math.exp(self.gate_log_temp)
        logits = []
        for c in range(8):
            center = self.centroids_[c]
            d = float(np.dot(vec, center))
            n1 = float(np.linalg.norm(vec))
            n2 = float(np.linalg.norm(center))
            sim = d / (n1 * n2 + 1e-10)
            logits.append(tau * sim + self.cluster_logits[c])
        peak = max(logits)
        exps = [math.exp(max(-30.0, min(30.0, v - peak))) for v in logits]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

    # ── three prediction heads ────────────────────────────────────────

    def _predict_fs(self, event, state):
        feats = [_features(event, o, state) for o in event.outcomes]
        scores = [_score(self.fs_weights, f) for f in feats]
        return _softmax(scores)

    def _predict_ms(self, event, state):
        feats = [_features(event, o, state) for o in event.outcomes]
        scores = [_score(self.ms_weights, f) for f in feats]
        # margin softmax: multiply each outcome's score by odds before softmax
        margin_scores = [s * o.odds for s, o in zip(scores, event.outcomes)]
        return _softmax(margin_scores)

    def _predict_v2(self, event, state):
        gate = self._gate_weights(event.main_title)
        feats = [_features(event, o, state) for o in event.outcomes]
        scores = [0.0] * len(event.outcomes)
        for c in range(8):
            w = self.cluster_weights[c]
            for j in range(len(event.outcomes)):
                scores[j] += gate[c] * _score(w, feats[j])
        return _softmax(scores)

    def _predict_ensemble(self, event, state):
        p1 = self._predict_fs(event, state)
        p2 = self._predict_ms(event, state)
        p3 = self._predict_v2(event, state)
        n = len(p1)
        return [(p1[i] + p2[i] + p3[i]) / 3.0 for i in range(n)]

    # ── inference ────────────────────────────────────────────────────

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        probs = self._predict_ensemble(event, state)
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
        probs = self._predict_ensemble(event, state)
        sel_idx = next(
            (i for i, o in enumerate(event.outcomes) if o.outcome_id == selected.outcome_id), None,
        )
        if sel_idx is None:
            return 0.0
        ordered = sorted(probs, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        edge = probs[sel_idx] * selected.odds - 1.0
        sig = _sigmoid(margin)
        amount = self.base_bet + self.edge_mult * max(0.0, edge) * sig
        return clamp(amount, 20000.0, 120000.0)

    # ── training (all three heads on each event) ──────────────────────

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        feats = [_features(event, o, state) for o in event.outcomes]
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]

        # Train feature_softmax
        fs_scores = [_score(self.fs_weights, f) for f in feats]
        fs_p = _softmax(fs_scores)
        fs_g = [p - yy for p, yy in zip(fs_p, y)]
        for i in range(N_FEATURES):
            g = sum(fs_g[j] * feats[j][FEATURE_NAMES[i]] for j in range(len(event.outcomes)))
            self.fs_weights[i] -= self.lr * g

        # Train margin_softmax (odds-weighted scores)
        ms_scores = [_score(self.ms_weights, f) for f in feats]
        ms_margin = [s * o.odds for s, o in zip(ms_scores, event.outcomes)]
        ms_p = _softmax(ms_margin)
        ms_g = [p - yy for p, yy in zip(ms_p, y)]
        for i in range(N_FEATURES):
            g = sum(ms_g[j] * feats[j][FEATURE_NAMES[i]] for j in range(len(event.outcomes)))
            self.ms_weights[i] -= self.lr * g

        # Train v2 cluster ensemble
        cidx = self._cluster_idx(event.main_title)
        w = self.cluster_weights[cidx]
        gate = self._gate_weights(event.main_title)
        v2_scores = [0.0] * len(event.outcomes)
        for c in range(8):
            for j in range(len(event.outcomes)):
                v2_scores[j] += gate[c] * _score(self.cluster_weights[c], feats[j])
        v2_p = _softmax(v2_scores)
        v2_g = [p - yy for p, yy in zip(v2_p, y)]
        for i in range(N_FEATURES):
            g = sum(v2_g[j] * feats[j][FEATURE_NAMES[i]] for j in range(len(event.outcomes)))
            w[i] -= self.lr * g

        # Gate update
        margin_loss = sum(abs(g) for g in v2_g) / len(v2_g) if v2_g else 0.0
        self.gate_log_temp += 0.01 * (1.0 - 2.0 * margin_loss)
        self.gate_log_temp = max(-3.0, min(10.0, self.gate_log_temp))
        for c in range(8):
            c_score = sum(_score(self.cluster_weights[c], feats[j]) * v2_g[j] for j in range(len(event.outcomes)))
            self.cluster_logits[c] += 0.002 * c_score
            self.cluster_logits[c] = max(-3.0, min(3.0, self.cluster_logits[c]))