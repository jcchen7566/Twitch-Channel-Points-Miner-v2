"""Omega 60k v10 — pure online global+residual with similarity-weighted ensemble.

Removes preload_history (no pretrain).  Instead:
  - Global weights (20 Feats, lr=0.05) updated EVERY event.
  - Every cluster residual updated per event, weighted by n-gram cosine sim.
  - CALIB_TEMP=2.5, cluster sizing multipliers, PnL modulation.

Hypothesis: ensemble backprop lets all clusters learn from every event,
which compensates for the missing multipass pretrain.
"""

import math
from collections import defaultdict

from ..core.strategy_base import StrategyBase
from ..core.features import build_features, clamp


# ── keyword cluster (same as omega v4/v5) ──────────────────────────

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
_CLUSTER_SIZE = [0.30, 0.20, 0.35, 1.30, 0.50, 0.90, 1.30, 0.30]
_CLUSTER_L2 = [1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]
_CALIB_TEMP = 2.5
_GLOBAL_LR = 0.05
_CLUSTER_LR = 0.005
# Seed titles for initial cluster centroid bootstrapping
_SEED_TITLES: dict[int, list[str]] = {
    0: ["BABA VS HENDERSON 誰贏"],
    1: ["大谷今天會失分嗎", "法官會開轟嗎"],
    2: ["這關命一關不刷會過嗎"],
    3: ["第幾名 改車 來賓開到幾星"],
    4: ["會不會過關"],
    5: ["最穩的那台"],
    6: ["丟骨頭"],
    7: ["等等要開哪種"],
}

_INITIAL_WEIGHTS = [
    -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
    0.00, 0.00, 0.08, -0.04, -0.02, 0.02,
    0.00, 0.00, 0.00, 0.00, 0.00,
    0.00,
]


class _ClusterCentroid:
    """Online n-gram centroid for similarity weighting."""
    __slots__ = ("ngram_sum", "count")

    def __init__(self):
        self.ngram_sum: dict[str, float] = defaultdict(float)
        self.count = 0

    def add(self, vec: dict[str, float]) -> None:
        for k, v in vec.items():
            self.ngram_sum[k] += v
        self.count += 1

    @property
    def centroid(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.ngram_sum.items()}

    def cosine(self, vec: dict[str, float]) -> float:
        cent = self.centroid
        if not cent or not vec:
            return 0.0
        dot = sum(v * cent.get(k, 0.0) for k, v in vec.items())
        n1 = math.sqrt(sum(v * v for v in vec.values()))
        n2 = math.sqrt(sum(v * v for v in cent.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)


def _ngrams(text: str) -> dict[str, float]:
    text = text.lower().strip()
    g: dict[str, float] = defaultdict(float)
    for n in (2, 3):
        for i in range(len(text) - n + 1):
            g[text[i:i + n]] += 1.0
    return dict(g)


# ── Strategy ────────────────────────────────────────────────────────

class Slrabbbit99Omega60kV10(StrategyBase):
    name = "slrabbit99_omega_60k_v10"

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

    def on_backtest_start(self):
        self.global_weights = list(_INITIAL_WEIGHTS)
        self.cluster_weights: dict[int, list[float]] = {
            c: [0.0] * len(self.FEATURE_NAMES) for c in range(N_CLUSTERS)
        }
        # Centroid for similarity weighting — seed then update online
        self.centroids: dict[int, _ClusterCentroid] = {
            c: _ClusterCentroid() for c in range(N_CLUSTERS)
        }
        for c, titles in _SEED_TITLES.items():
            for t in titles:
                v = _ngrams(t)
                if any(x > 0 for x in v.values()):
                    self.centroids[c].add(v)

        self._streak = 0
        self._pnl_window: list[float] = []

    # ── helpers ──

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
            "liq_low": 1.0 if 0 < tp < 500000 else 0.0,
            "liq_mid": 1.0 if 500000 <= tp <= 2000000 else 0.0,
            "odds_high": 1.0 if outcome.odds > 1.8 else 0.0,
            "whale": 1.0 if pp > 0 and (up / pp) < 0.9 else 0.0,
            "streak": self._streak / 10.0,
        }

    @staticmethod
    def _score_outcome(weights, feat):
        return sum(w * feat[n] for w, n in zip(weights, [
            "bias", "log_odds", "users_pct", "points_pct", "ratio",
            "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
            "p", "ev", "crowd_liq",
            "users_minus_points", "points_minus_users", "n_out",
            "liq_low", "liq_mid", "odds_high", "whale",
            "streak",
        ]))

    def _ensemble_weights(self, title: str) -> list[tuple[int, float]]:
        """Return [(cid, normalized_weight)] for all clusters."""
        vec = _ngrams(title)
        raw = []
        for c in range(N_CLUSTERS):
            sim = max(0.01, self.centroids[c].cosine(vec))
            raw.append(sim)
        total = sum(raw)
        return [(c, s / total) for c, s in enumerate(raw)]

    def _ensemble_scores(self, event, state):
        title = event.main_title
        feats = [self._features(event, o, state) for o in event.outcomes]
        ew = self._ensemble_weights(title)

        scores = [0.0] * len(event.outcomes)
        for j in range(len(event.outcomes)):
            s = self._score_outcome(self.global_weights, feats[j])
            for c, w in ew:
                s += w * self._score_outcome(self.cluster_weights[c], feats[j])
            scores[j] = s

        calibrated = [s / _CALIB_TEMP for s in scores]
        probs = self._softmax(calibrated)
        return feats, scores, probs, ew

    # ── inference ──

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
        _, scores, probs, ew = self._ensemble_scores(event, state)
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

        # Cluster sizing multiplier
        cid = _keyword_cluster(event.main_title)
        amount *= _CLUSTER_SIZE[cid]

        # PnL modulation
        n_pnl = len(self._pnl_window)
        if n_pnl >= 10:
            mean_pnl = sum(self._pnl_window) / n_pnl
            var_pnl = sum((x - mean_pnl) ** 2 for x in self._pnl_window) / n_pnl
            std_pnl = math.sqrt(var_pnl) if var_pnl > 0 else 1e-6
            sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
            amount *= 1.0 + 0.4 * max(-0.8, min(0.8, sharpe))

        return clamp(amount, 5000.0, 150000.0)

    # ── online training ──

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        self._pnl_window.append(pnl)
        if len(self._pnl_window) > 30:
            self._pnl_window.pop(0)

        if pnl > 0:
            self._streak = min(10, self._streak + 1)
        else:
            self._streak = max(-10, self._streak - 1)

        feats, scores, probs, ew = self._ensemble_scores(event, state)
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]  # dL/d(calibrated_score)

        # ── Update global weights (full gradient, high LR) ──
        for i, name in enumerate(self.FEATURE_NAMES):
            g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
            self.global_weights[i] -= _GLOBAL_LR * g

        # ── Update ALL cluster residuals (weighted by ensemble weight) ──
        for c, w in ew:
            cw = self.cluster_weights[c]
            l2 = _CLUSTER_L2[c]
            for i, name in enumerate(self.FEATURE_NAMES):
                g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
                cw[i] -= _CLUSTER_LR * w * (g + l2 * cw[i])

        # ── Update best-matching cluster centroid (cumulative) ──
        cid = _keyword_cluster(event.main_title)
        self.centroids[cid].add(_ngrams(event.main_title))
