"""v13 — v1 architecture + Polyak-averaged weights (self-distillation).

Maintains slow (EMA) weights for prediction while training fast weights:
  slow = 0.999 × slow + 0.001 × fast  (per update)
  predictions use slow weights → smoother, better generalization.

Training loss uses fast weights (standard cross-entropy).
Only inference uses slow weights.
"""
import math
from ..core.strategy_base import StrategyBase
from ..core.features import build_features, clamp


def _keyword_cluster(title: str) -> int:
    t = (title or "").strip().lower()
    if any(k in t for k in ["誰會贏", "哪邊會贏", " vs ", "誰贏"]): return 0
    if any(k in t for k in ["失分", "勝投", "全壘打", "安打", "觸身", "er ", "er)"]): return 1
    if any(k in t for k in ["命一關不刷", "一關不刷"]): return 2
    if any(k in t for k in ["第幾名", "開到幾星", "抽"]): return 3
    if any(k in t for k in ["會不會過關", "能不能過關", "一把決勝負"]): return 4
    if "最穩的那台" in t: return 5
    if "丟骨頭" in t: return 6
    return 7


N_CLUSTERS = 8
LR = 0.08
POLYAK_DECAY = 0.995

FEATURES = [
    "bias", "log_odds", "users_pct", "points_pct", "ratio",
    "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
    "p", "ev", "conf", "momentum", "crowd_liq",
    "users_minus_points", "points_minus_users", "efficiency",
]

_INITIAL_WEIGHTS = [
    -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
    0.00, 0.00, 0.08, 0.06, 0.00, -0.04, -0.02, 0.02, 0.00,
]


class Slrabbit99ClusterEnsembleV13(StrategyBase):
    name = "slrabbit99_cluster_ensemble_v13"

    def on_backtest_start(self):
        self.fast = {c: list(_INITIAL_WEIGHTS) for c in range(N_CLUSTERS)}
        self.slow = {c: list(_INITIAL_WEIGHTS) for c in range(N_CLUSTERS)}
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
        return {
            "bias": 1.0, "log_odds": f["log_odds"],
            "users_pct": outcome.users_pct / 100.0, "points_pct": outcome.points_pct / 100.0,
            "ratio": f["ratio"], "rel_ratio": f["rel_ratio"],
            "rel_odds": f["rel_odds"], "rel_surprise": f["rel_surprise"],
            "log_liq": f["log_liq"], "p": f["p"], "ev": f["ev"],
            "conf": f["conf"], "momentum": f["momentum"],
            "crowd_liq": f["crowd_liq"] / 10.0,
            "users_minus_points": (outcome.users_pct - outcome.points_pct) / 100.0,
            "points_minus_users": (outcome.points_pct - outcome.users_pct) / 100.0,
            "efficiency": f["efficiency"],
        }

    @staticmethod
    def _score(w, f):
        return sum(w[i] * f[n] for i, n in enumerate(FEATURES))

    def _ensemble(self, event, state, use_slow=True):
        weights = self.slow if use_slow else self.fast
        cidx = _keyword_cluster(event.main_title)
        feats = [self._features(event, o, state) for o in event.outcomes]
        scores = [0.0] * len(event.outcomes)
        for c in range(N_CLUSTERS):
            w = weights[c]
            sim = 0.6 if c == cidx else 0.4 / (N_CLUSTERS - 1)
            for j in range(len(event.outcomes)):
                scores[j] += sim * self._score(w, feats[j])
        probs = self._softmax(scores)
        return feats, scores, probs, cidx

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        _, _, probs, _ = self._ensemble(event, state, use_slow=True)
        best, best_edge = None, -10**9
        for o, p in zip(event.outcomes, probs):
            if (e := p * o.odds - 1.0) > best_edge:
                best_edge = e
                best = o
        return best

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99" or selected is None:
            return 0.0
        _, scores, probs, _ = self._ensemble(event, state, use_slow=True)
        sel_idx = next((i for i, o in enumerate(event.outcomes) if o.outcome_id == selected.outcome_id), None)
        if sel_idx is None:
            return 0.0
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        p = probs[sel_idx]
        edge = p * selected.odds - 1.0
        amount = 20000 + 100000 * max(0.0, edge) * (1.0 / (1.0 + math.exp(-margin)) if margin < 35 else 1.0)
        return clamp(amount, 20000.0, 120000.0)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return
        self._pnl_window.append(pnl)
        if len(self._pnl_window) > 30:
            self._pnl_window.pop(0)
        self._streak = min(10, self._streak + 1) if pnl > 0 else max(-10, self._streak - 1)

        cidx = _keyword_cluster(event.main_title)
        feats, scores, probs, _ = self._ensemble(event, state, use_slow=False)
        y = [1.0 if o.outcome_id == event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p - yy for p, yy in zip(probs, y)]
        fw = self.fast[cidx]
        sw = self.slow[cidx]

        for i, name in enumerate(FEATURES):
            g = sum(grad[j] * feats[j][name] for j in range(len(event.outcomes)))
            fw[i] -= LR * g
            sw[i] = POLYAK_DECAY * sw[i] + (1 - POLYAK_DECAY) * fw[i]
