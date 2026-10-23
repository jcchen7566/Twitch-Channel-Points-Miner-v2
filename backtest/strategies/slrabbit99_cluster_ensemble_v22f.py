"""v22f — v22d + CALIB_TEMP=0.8 (sharper probs, favors longshots)."""
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
FEATURES = ["bias", "log_odds", "users_pct", "points_pct", "ratio", "rel_ratio", "rel_odds", "rel_surprise", "log_liq", "p", "ev", "conf", "momentum", "crowd_liq", "users_minus_points", "points_minus_users", "efficiency"]
_INITIAL_WEIGHTS = [-0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06, 0.00, 0.00, 0.08, 0.06, 0.00, -0.04, -0.02, 0.02, 0.00]
BASE_LR = 0.09
RMS_DECAY = 0.85
CALIB_TEMP = 0.8

class Slrabbit99ClusterEnsembleV22f(StrategyBase):
    name = "slrabbit99_cluster_ensemble_v22f"
    def on_backtest_start(self):
        self.cluster_weights = {c: list(_INITIAL_WEIGHTS) for c in range(N_CLUSTERS)}
        self._sq = {c: [0.01]*len(FEATURES) for c in range(N_CLUSTERS)}
        self._cluster_n = {c: 0.0 for c in range(N_CLUSTERS)}

    def _lr(self, cidx):
        n = max(5.0, self._cluster_n[cidx])
        return BASE_LR * math.sqrt(73.0 / n)

    @staticmethod
    def _calib_softmax(values):
        if not values: return []
        scaled = [v / CALIB_TEMP for v in values]
        peak = max(scaled)
        exps = [math.exp(max(-20.0, min(20.0, v - peak))) for v in scaled]
        d = sum(exps) or 1.0; return [v / d for v in exps]

    def _features(self, event, outcome, state):
        f = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        return dict(bias=1.0, log_odds=f["log_odds"], users_pct=outcome.users_pct/100.0, points_pct=outcome.points_pct/100.0, ratio=f["ratio"], rel_ratio=f["rel_ratio"], rel_odds=f["rel_odds"], rel_surprise=f["rel_surprise"], log_liq=f["log_liq"], p=f["p"], ev=f["ev"], conf=f["conf"], momentum=f["momentum"], crowd_liq=f["crowd_liq"]/10.0, users_minus_points=(outcome.users_pct-outcome.points_pct)/100.0, points_minus_users=(outcome.points_pct-outcome.users_pct)/100.0, efficiency=f["efficiency"])

    @staticmethod
    def _score(w, f): return sum(w[i]*f[n] for i,n in enumerate(FEATURES))

    def _ensemble(self, event, state):
        cidx = _keyword_cluster(event.main_title)
        feats = [self._features(event, o, state) for o in event.outcomes]
        scores = [0.0]*len(event.outcomes)
        for c in range(N_CLUSTERS):
            w = self.cluster_weights[c]; sim = 0.6 if c==cidx else 0.4/(N_CLUSTERS-1)
            for j in range(len(event.outcomes)): scores[j] += sim*self._score(w, feats[j])
        return feats, scores, self._calib_softmax(scores), cidx

    def pick(self, event, state):
        if event.streamer != "slrabbit99": return None
        _,_,probs,_ = self._ensemble(event, state)
        best, best_e = None, -10**9
        for o,p in zip(event.outcomes, probs):
            if (e:=p*o.odds-1.0) > best_e: best_e, best = e, o
        return best

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99" or selected is None: return 0.0
        _,scores,probs,_ = self._ensemble(event, state)
        sel_idx = next((i for i,o in enumerate(event.outcomes) if o.outcome_id==selected.outcome_id), None)
        if sel_idx is None: return 0.0
        ordered = sorted(scores, reverse=True)
        margin = ordered[0]-ordered[1] if len(ordered)>1 else ordered[0]
        edge = probs[sel_idx]*selected.odds-1.0
        return clamp(20000+100000*max(0.0,edge)*(1.0/(1.0+math.exp(-margin)) if margin<35 else 1.0), 20000.0, 120000.0)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99": return
        cidx = _keyword_cluster(event.main_title)
        self._cluster_n[cidx] += 1.0
        feats,_,probs,_ = self._ensemble(event, state)
        y = [1.0 if o.outcome_id==event.winner_outcome_id else 0.0 for o in event.outcomes]
        grad = [p-yy for p,yy in zip(probs,y)]
        w = self.cluster_weights[cidx]; sq = self._sq[cidx]; lr = self._lr(cidx)
        for i,name in enumerate(FEATURES):
            g = sum(grad[j]*feats[j][name] for j in range(len(event.outcomes)))
            sq[i] = RMS_DECAY*sq[i] + (1-RMS_DECAY)*g*g
            w[i] -= lr*g / (math.sqrt(sq[i]) + 1e-6)
