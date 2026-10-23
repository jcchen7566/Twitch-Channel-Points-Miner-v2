"""Omega 60k v10b — pure online, v1 ensemble + omega's 20 features + CALIB_TEMP.

Hypothesis: v1's 0.6/0.4 ensemble works at 28%.  Upgrading to 20 features
(n_out, liq_low/mid, odds_high, whale, streak) + CALIB_TEMP=2.5 + cluster
sizing multipliers + PnL modulation should push beyond v1's 28%.
"""
import math
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


N_CLUSTERS = 8
_CLUSTER_SIZE = [0.30, 0.20, 0.35, 1.30, 0.50, 0.90, 1.30, 0.30]
_CALIB_TEMP = 1.0
_LR = 0.05

_INITIAL_WEIGHTS = [
    -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
    0.00, 0.00, 0.08, -0.04, -0.02, 0.02,
    0.00, 0.00, 0.00, 0.00, 0.00,
    0.00,
]
FEATURES = [
    "bias", "log_odds", "users_pct", "points_pct", "ratio",
    "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
    "p", "ev", "crowd_liq",
    "users_minus_points", "points_minus_users", "n_out",
    "liq_low", "liq_mid", "odds_high", "whale",
    "streak",
]


class Slrabbbit99Omega60kV10b(StrategyBase):
    name = "slrabbit99_omega_60k_v10b"

    def on_backtest_start(self):
        self.cluster_weights = {c: list(_INITIAL_WEIGHTS) for c in range(N_CLUSTERS)}
        self._streak = 0
        self._pnl_window: list[float] = []

    @staticmethod
    def _softmax(values):
        if not values:
            return []
        peak = max(values)
        exps = [math.exp(max(-20.0, min(20.0, v - peak))) for v in values]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

    def _features(self, event, outcome, state):
        f = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        tp, up, pp = outcome.total_points, outcome.users_pct, outcome.points_pct
        return {
            "bias": 1.0, "log_odds": f["log_odds"],
            "users_pct": up / 100.0, "points_pct": pp / 100.0,
            "ratio": f["ratio"], "rel_ratio": f["rel_ratio"],
            "rel_odds": f["rel_odds"], "rel_surprise": f["rel_surprise"],
            "log_liq": f["log_liq"], "p": f["p"], "ev": f["ev"],
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
    def _score(w, f):
        return sum(w[i] * f[n] for i, n in enumerate(FEATURES))

    def _ensemble(self, event, state):
        cidx = _keyword_cluster(event.main_title)
        feats = [self._features(event, o, state) for o in event.outcomes]
        scores = [0.0] * len(event.outcomes)
        for c in range(N_CLUSTERS):
            w = self.cluster_weights[c]
            sim = 0.6 if c == cidx else 0.4 / (N_CLUSTERS - 1)
            for j in range(len(event.outcomes)):
                scores[j] += sim * self._score(w, feats[j])
        cal = [s / _CALIB_TEMP for s in scores]
        probs = self._softmax(cal)
        return feats, scores, probs, cidx

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        _, _, probs, _ = self._ensemble(event, state)
        best, best_edge = None, -10**9
        for o, p in zip(event.outcomes, probs):
            if (e := p * o.odds - 1.0) > best_edge:
                best_edge = e
                best = o
        return best

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99" or selected is None:
            return 0.0
        _, scores, probs, _ = self._ensemble(event, state)
        sel_idx = next((i for i, o in enumerate(event.outcomes) if o.outcome_id == selected.outcome_id), None)
        if sel_idx is None:
            return 0.0
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        p = probs[sel_idx]
        edge = p * selected.odds - 1.0
        sig = 1.0 / (1.0 + math.exp(-margin)) if margin < 35 else 1.0
        amount = 100 + 100000 * max(0.0, edge) * sig
        amount *= _CLUSTER_SIZE[_keyword_cluster(event.main_title)]
        n = len(self._pnl_window)
        if n >= 10:
            m = sum(self._pnl_window) / n
            v = sum((x - m)**2 for x in self._pnl_window) / n
            std = math.sqrt(v) if v > 0 else 1e-6
            amount *= 1.0 + 0.4 * max(-0.8, min(0.8, m / std))
        return clamp(amount, 5000.0, 150000.0)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return
        self._pnl_window.append(pnl)
        if len(self._pnl_window) > 30:
            self._pnl_window.pop(0)
        self._streak = min(10, self._streak + 1) if pnl > 0 else max(-10, self._streak - 1)
        cidx = _keyword_cluster(event.main_title)
        feats, scores, probs, _ = self._ensemble(event, state)
        grad = [p - (1.0 if o.outcome_id == event.winner_outcome_id else 0.0) for p, o in zip(probs, event.outcomes)]
        w = self.cluster_weights[cidx]
        for i, name in enumerate(FEATURES):
            w[i] -= _LR * sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
