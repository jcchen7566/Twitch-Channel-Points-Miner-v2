"""v8 — v1 pure online + momentum + CALIB_TEMP + cluster sizing.

v1 proven architecture (123% WF) with:
  - momentum (0.9) on weight updates (faster convergence)
  - CALIB_TEMP=2.5 (prevents overconfident probs)
  - cluster sizing multipliers from v4
  - PnL modulation via windowed Sharpe
"""

import math

from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase


def _cluster_idx(title: str) -> int:
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
    "WHO_WIN", "BASEBALL", "LIFE_GAMES", "NUMBER_GUESS",
    "CHALLENGE", "LADDER", "BONE", "CATCH_ALL",
]
N_CLUSTERS = len(CLUSTER_NAMES)
_CALIB_TEMP = 2.5
_CLUSTER_SIZE = [0.30, 0.20, 0.35, 1.30, 0.50, 0.90, 1.30, 0.30]


class Slrabbbit99Omega60kV8(StrategyBase):
    name = "slrabbbit99_omega_60k_v8"

    FEATURE_NAMES = [
        "bias", "log_odds", "users_pct", "points_pct", "ratio",
        "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
        "p", "ev", "conf", "momentum", "crowd_liq",
        "users_minus_points", "points_minus_users", "efficiency",
    ]

    def __init__(self):
        super().__init__()
        self.cluster_weights = {}
        self.cluster_velocities = {}

    def on_backtest_start(self):
        self.lr = 0.05
        self.momentum = 0.9
        self.base_bet = 20000.0
        self.edge_mult = 100000.0
        self._initial_weights = [
            -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
            0.00, 0.00, 0.08, 0.06, 0.00, -0.04, -0.02, 0.02, 0.00,
        ]
        self.cluster_weights = {
            c: list(self._initial_weights) for c in range(N_CLUSTERS)
        }
        self.cluster_velocities = {
            c: [0.0] * len(self._initial_weights) for c in range(N_CLUSTERS)
        }
        self._pnl_window = []

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
            "conf": f["conf"],
            "momentum": f["momentum"],
            "crowd_liq": f["crowd_liq"] / 10.0,
            "users_minus_points": (up - pp) / 100.0,
            "points_minus_users": (pp - up) / 100.0,
            "efficiency": f["efficiency"],
        }

    def _score_outcome(self, weights, feat):
        return sum(w * feat[name] for w, name in zip(weights, self.FEATURE_NAMES))

    def _ensemble_scores(self, event, state):
        cidx = _cluster_idx(event.main_title)
        feats = [self._features(event, o, state) for o in event.outcomes]
        scores = [0.0] * len(event.outcomes)
        for c in range(N_CLUSTERS):
            w = self.cluster_weights[c]
            sim = 0.6 if c == cidx else 0.4 / (N_CLUSTERS - 1 + 1e-10)
            for j in range(len(event.outcomes)):
                scores[j] += sim * self._score_outcome(w, feats[j])
        calibrated = [s / _CALIB_TEMP for s in scores]
        probs = self._softmax(calibrated)
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
        sig = 1.0 / (1.0 + math.exp(-margin)) if margin < 35 else 1.0
        amount = self.base_bet + self.edge_mult * max(0.0, edge) * sig

        cidx = _cluster_idx(event.main_title)
        amount *= _CLUSTER_SIZE[cidx]

        n_pnl = len(self._pnl_window)
        if n_pnl >= 10:
            mean_pnl = sum(self._pnl_window) / n_pnl
            var_pnl = sum((x - mean_pnl) ** 2 for x in self._pnl_window) / n_pnl
            std_pnl = math.sqrt(var_pnl) if var_pnl > 0 else 1e-6
            sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
            mod = 1.0 + 0.4 * max(-0.8, min(0.8, sharpe))
            amount *= mod

        return clamp(amount, 5000.0, 150000.0)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        self._pnl_window.append(pnl)
        if len(self._pnl_window) > 30:
            self._pnl_window.pop(0)

        cidx = _cluster_idx(event.main_title)
        w = self.cluster_weights[cidx]
        v = self.cluster_velocities[cidx]

        feats, scores, probs, _ = self._ensemble_scores(event, state)
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]

        for i, name in enumerate(self.FEATURE_NAMES):
            g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
            v[i] = self.momentum * v[i] + self.lr * g
            w[i] -= v[i]
