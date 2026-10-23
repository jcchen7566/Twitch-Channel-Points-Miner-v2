"""Omega 60k v6 — hybrid fixed+dynamic clustering, frozen global weights.

v5 failure analysis:
  - KMeans alone destroyed semantic cluster boundaries
  - Global online SGD corrupted pretrained weights (compound drift)
  - New clusters started at zero — terrible cold-start predictions
  - Inverted sizing (more weight on smaller clusters -> bet big on noise)

v6 fixes:
  - Hybrid: keyword-based for 8 known clusters + dynamic fallback for novel titles
  - Global weights FROZEN after pretrain (no online SGD)
  - Dynamic clusters inherit weights from nearest keyword cluster (not zero)
  - EMA centroid only for dynamic clusters (keyword centroids fixed)
  - Empirical sizing multipliers from v4 for keyword clusters
  - Data-driven sizing for dynamic clusters (shrinkage toward 1.0)
  - Prevent cluster over-creation (>3 novel clusters max)
"""

import math
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from ..core.strategy_base import StrategyBase
from ..core.features import build_features, clamp


# ---- keyword cluster definitions (same as v4) ----
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
    return 7  # CATCH_ALL

KW_CLUSTER_NAMES = [
    "WHO_WIN", "BASEBALL", "LIFE_GAMES", "NUMBER_GUESS",
    "CHALLENGE", "LADDER", "BONE", "CATCH_ALL",
]
N_KW_CLUSTERS = len(KW_CLUSTER_NAMES)
KW_CLUSTER_LABELS = list(range(N_KW_CLUSTERS))

# v4-validated sizing multipliers
_KW_SIZE = [0.30, 0.20, 0.35, 1.30, 0.50, 0.90, 1.30, 0.30]
_KW_L2 = [1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]

# --- global tunables ---
_CALIB_TEMP = 2.5
_PRETRAIN_LR = 0.06
_PRETRAIN_EPOCHS_GLOBAL = 10
_PRETRAIN_EPOCHS_CLUSTER = 8
_MAX_DYNAMIC_CLUSTERS = 3
_DYN_INHERIT_WEIGHT = 0.3  # blending: (1-w)*parent + w*zero  (init near parent)
_DYN_NEW_CLUSTER_THRESH = 0.35  # cosine distance to create new dynamic cluster


def _cosine_sim(a, b):
    n1 = float(np.linalg.norm(a))
    n2 = float(np.linalg.norm(b))
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(a, b)) / (n1 * n2)


class Slrabbbit99Omega60kV5(StrategyBase):
    name = "slrabbit99_omega_60k_v5"

    FEATURE_NAMES = [
        "bias", "log_odds", "users_pct", "points_pct", "ratio",
        "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
        "p", "ev", "crowd_liq",
        "users_minus_points", "points_minus_users", "n_out",
        "liq_low", "liq_mid", "odds_high", "whale",
        "streak",
    ]

    def __init__(self, base_bet=100, edge_mult=100000):
        super().__init__()
        self.base_bet = base_bet
        self.edge_mult = edge_mult
        self.gate_log_temp = math.log(8.0)
        self.global_weights = []
        self.cluster_weights = {}       # { cid: [weights] }
        self.cluster_sizes = {}          # { cid: int }
        self.cluster_multipliers = {}    # { cid: float }
        self._fitted = False
        self._streak = 0
        self._pnl_window = []
        # dynamic cluster tracking
        self._dyn_start_id = N_KW_CLUSTERS  # first dynamic cluster ID
        self._n_dynamic = 0

    def on_backtest_start(self):
        self.gate_log_temp = math.log(8.0)
        self._streak = 0
        self._pnl_window = []

        self._initial_weights = [
            -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
            0.00, 0.00, 0.08, -0.04, -0.02, 0.02,
            0.00, 0.00, 0.00, 0.00, 0.00,
            0.00,
        ]

        if not self.global_weights:
            self.global_weights = list(self._initial_weights)
        # Reset cluster state (fresh for each backtest)
        self.cluster_weights = {}
        self.cluster_sizes = {}
        self.cluster_multipliers = {}
        self._n_dynamic = 0

    # ---- helpers ----

    @staticmethod
    def _softmax(values):
        if not values:
            return []
        peak = max(values)
        exps = [math.exp(max(-20.0, min(20.0, v - peak))) for v in values]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

    def _embed_title(self, title: str) -> np.ndarray:
        X = self.vectorizer.transform([title])
        return self.svd.transform(X)[0]

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

    # ---- pretrain ----

    def preload_history(self, events):
        titles = [e.main_title for e in events if e.streamer == "slrabbit99" and e.main_title]
        titles = list(dict.fromkeys(titles))
        if not titles:
            return

        # Build embedding for gating
        self.vectorizer = CountVectorizer(
            analyzer="char_wb", ngram_range=(2, 3),
            max_features=2000, binary=True,
        )
        n_dims = min(20, len(titles) - 1)
        self.svd = TruncatedSVD(n_components=max(1, n_dims), random_state=42)
        X = self.vectorizer.fit_transform(titles)
        Xr = self.svd.fit_transform(X)

        # Build centroid per keyword cluster (from pretrain data)
        self.kw_centroids = {}
        for c in range(N_KW_CLUSTERS):
            c_titles = [t for i, t in enumerate(titles) if _keyword_cluster(t) == c]
            if c_titles:
                c_vecs = [Xr[titles.index(t)] for t in c_titles]
                self.kw_centroids[c] = np.mean(c_vecs, axis=0)
            else:
                self.kw_centroids[c] = np.zeros(n_dims)

        self._n_dims = n_dims
        self._fitted = True

        sl_events = [e for e in events if e.streamer == "slrabbit99" and e.main_title]
        mock = type("Mock", (), {"prior_win_rate": lambda s, st, od: 0.5})()

        # ---- Pretrain global model (frozen after this) ----
        gw = list(self._initial_weights)
        for _ in range(_PRETRAIN_EPOCHS_GLOBAL):
            for e in sl_events:
                feats = [self._features(e, o, mock) for o in e.outcomes]
                scores = [self._score_outcome(gw, f) for f in feats]
                cal = [s / _CALIB_TEMP for s in scores]
                probs = self._softmax(cal)
                grad = [p - (1.0 if o.outcome_id == e.winner_outcome_id else 0.0)
                        for p, o in zip(probs, e.outcomes)]
                for i in range(len(gw)):
                    g = sum(grad[j] * feats[j][self.FEATURE_NAMES[i]]
                            for j in range(len(e.outcomes)))
                    gw[i] -= _PRETRAIN_LR * g
        self.global_weights = gw  # FROZEN — no online updates

        # ---- Pretrain per-cluster residuals (keyword clusters only) ----
        self.cluster_weights = {}
        self.cluster_sizes = {}
        self.cluster_multipliers = {}

        for c in range(N_KW_CLUSTERS):
            ce = [e for e in sl_events if _keyword_cluster(e.main_title) == c]
            self.cluster_sizes[c] = len(ce)
            self.cluster_multipliers[c] = _KW_SIZE[c]
            if len(ce) < 3:
                self.cluster_weights[c] = [0.0] * len(gw)
                continue
            cw = [0.0] * len(gw)
            for _ in range(_PRETRAIN_EPOCHS_CLUSTER):
                for e in ce:
                    feats = [self._features(e, o, mock) for o in e.outcomes]
                    total = [self._score_outcome(gw, f) + self._score_outcome(cw, f) for f in feats]
                    cal = [s / _CALIB_TEMP for s in total]
                    probs = self._softmax(cal)
                    grad = [p - (1.0 if o.outcome_id == e.winner_outcome_id else 0.0)
                            for p, o in zip(probs, e.outcomes)]
                    for i in range(len(cw)):
                        g = sum(grad[j] * feats[j][self.FEATURE_NAMES[i]]
                                for j in range(len(e.outcomes)))
                        cw[i] -= _PRETRAIN_LR * 0.6 * g
            self.cluster_weights[c] = cw

    # ---- gating ----

    def _resolve_cluster(self, title: str) -> int:
        """Return cluster_id via keyword match; dynamic fallback only for CATCH_ALL novel."""
        if not title:
            return 0

        kw_c = _keyword_cluster(title)
        if kw_c < 7:
            return kw_c

        # CATCH_ALL (7) — create dynamic sub-cluster only if this title is truly novel
        # and can't be represented by the existing CATCH_ALL centroid.
        if not hasattr(self, 'dyn_centroids'):
            self.dyn_centroids = {}
        if not hasattr(self, 'dyn_cluster_for_title'):
            self.dyn_cluster_for_title = {}

        # Already assigned to a dynamic cluster?
        if title in self.dyn_cluster_for_title:
            return self.dyn_cluster_for_title[title]

        # Check embedding distance to existing dynamic clusters
        vec = self._embed_title(title)
        for c, center in list(self.dyn_centroids.items()):
            sim = _cosine_sim(vec, center)
            if sim >= _DYN_NEW_CLUSTER_THRESH:
                self.dyn_cluster_for_title[title] = c
                return c

        # Check against CATCH_ALL centroid
        catchall_center = self.kw_centroids.get(7, np.zeros(self._n_dims))
        sim_to_catchall = _cosine_sim(vec, catchall_center)

        if sim_to_catchall < _DYN_NEW_CLUSTER_THRESH and self._n_dynamic < _MAX_DYNAMIC_CLUSTERS:
            new_id = self._dyn_start_id + self._n_dynamic
            self.dyn_centroids[new_id] = vec.copy()
            self.dyn_cluster_for_title[title] = new_id
            parent_w = self.cluster_weights.get(7, [0.0] * len(self.FEATURE_NAMES))
            self.cluster_weights[new_id] = [(1.0 - _DYN_INHERIT_WEIGHT) * pw for pw in parent_w]
            self.cluster_sizes[new_id] = 0
            self.cluster_multipliers[new_id] = 1.0
            self._n_dynamic += 1
            return new_id

        return 7

    # ---- ensemble (no global online updates) ----

    def _ensemble_scores(self, event, state):
        cid = self._resolve_cluster(event.main_title)
        feats = [self._features(event, o, state) for o in event.outcomes]
        scores = [0.0] * len(event.outcomes)

        # Global base (frozen)
        for j in range(len(event.outcomes)):
            scores[j] = self._score_outcome(self.global_weights, feats[j])

        # Cluster residual (best_only)
        if cid in self.cluster_weights:
            w = self.cluster_weights[cid]
            for j in range(len(event.outcomes)):
                scores[j] += self._score_outcome(w, feats[j])

        calibrated = [s / _CALIB_TEMP for s in scores]
        probs = self._softmax(calibrated)
        return feats, scores, probs, cid

    # ---- inference ----

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
        _, scores, probs, cid = self._ensemble_scores(event, state)
        sel_idx = next(
            (i for i, o in enumerate(event.outcomes) if o.outcome_id == selected.outcome_id), None,
        )
        if sel_idx is None:
            return 0.0
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        p = probs[sel_idx]
        edge = p * selected.odds - 1.0
        sig = 1.0 / (1.0 + math.exp(-margin)) if margin < 35 else 1.0
        amount = self.base_bet + self.edge_mult * max(0.0, edge) * sig

        # Cluster sizing multiplier (empirical for KW, data-driven for dynamic)
        mult = self.cluster_multipliers.get(cid, 1.0)
        if cid >= self._dyn_start_id:
            # Dynamic clusters: shrink toward 1.0 based on data count
            cnt = max(self.cluster_sizes.get(cid, 1), 1)
            mult = 1.0 + (mult - 1.0) * min(1.0, cnt / 10.0)
        amount *= mult

        # PnL modulation
        n_pnl = len(self._pnl_window)
        if n_pnl >= 10:
            mean_pnl = sum(self._pnl_window) / n_pnl
            var_pnl = sum((x - mean_pnl) ** 2 for x in self._pnl_window) / n_pnl
            std_pnl = math.sqrt(var_pnl) if var_pnl > 0 else 1e-6
            sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
            amount *= 1.0 + 0.4 * max(-0.8, min(0.8, sharpe))

        return clamp(amount, 5000.0, 150000.0)

    # ---- online training (cluster weights + centroid EMA only, NO global) ----

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        self._pnl_window.append(pnl)
        if len(self._pnl_window) > 30:
            self._pnl_window.pop(0)

        cid = self._resolve_cluster(event.main_title)

        if pnl > 0:
            self._streak = min(10, self._streak + 1)
        else:
            self._streak = max(-10, self._streak - 1)

        feats, scores, probs, _ = self._ensemble_scores(event, state)
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]

        # --- Update cluster weights (NO global updates) ---
        if cid in self.cluster_weights:
            cw = self.cluster_weights[cid]
            l2 = _KW_L2[cid] if cid < N_KW_CLUSTERS else 1e-5
            lr = 0.005
            for i, name in enumerate(self.FEATURE_NAMES):
                g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
                cw[i] -= lr * (g + l2 * cw[i])

        # --- Update cluster size counter ---
        self.cluster_sizes[cid] = self.cluster_sizes.get(cid, 0) + 1

        # --- Update dynamic cluster centroid via EMA ---
        if cid >= self._dyn_start_id and hasattr(self, 'dyn_centroids') and cid in self.dyn_centroids:
            vec = self._embed_title(event.main_title)
            alpha = 0.15
            self.dyn_centroids[cid] = (1.0 - alpha) * self.dyn_centroids[cid] + alpha * vec

        # --- Update dynamic cluster multiplier based on performance ---
        if cid >= self._dyn_start_id:
            # Track win rate for dynamic clusters
            if not hasattr(self, '_dyn_pnl'):
                self._dyn_pnl = {}
            self._dyn_pnl.setdefault(cid, []).append(1.0 if won else 0.0)
            recent = self._dyn_pnl[cid][-20:]
            if len(recent) >= 5:
                wr = sum(recent) / len(recent)
                # Adjust multiplier: WR > 0.5 → increase, WR < 0.5 → decrease, capped
                adj = 1.0 + 0.5 * (wr - 0.5) * 2.0  # 0.5->1.0, 0.75->1.25, 0.25->0.75
                self.cluster_multipliers[cid] = clamp(adj, 0.3, 2.0)

        # --- Gate log temp ---
        margin_loss = sum(abs(g) for g in grad) / len(grad) if grad else 0.0
        self.gate_log_temp += 0.005 * (1.0 - 2.0 * margin_loss)
        self.gate_log_temp = max(-3.0, min(10.0, self.gate_log_temp))
