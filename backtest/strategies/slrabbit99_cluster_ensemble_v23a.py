"""v23a — v22d + Kelly sizing: size = min + (edge/(odds-1)) * balance_scalar."""
import math
from ..core.strategy_base import StrategyBase
from ..core.features import build_features, clamp

def _keyword_cluster(title: str) -> int:
    t = (title or "").strip().lower()
    if any(k in t for k in ["誰會贏","哪邊會贏"," vs ","誰贏"]): return 0
    if any(k in t for k in ["失分","勝投","全壘打","安打","觸身","er ","er)"]): return 1
    if any(k in t for k in ["命一關不刷","一關不刷"]): return 2
    if any(k in t for k in ["第幾名","開到幾星","抽"]): return 3
    if any(k in t for k in ["會不會過關","能不能過關","一把決勝負"]): return 4
    if "最穩的那台" in t: return 5
    if "丟骨頭" in t: return 6
    return 7

N_CLUSTERS=8; BASE_LR=0.09; RMS_DECAY=0.85
FEATURES=["bias","log_odds","users_pct","points_pct","ratio","rel_ratio","rel_odds","rel_surprise","log_liq","p","ev","conf","momentum","crowd_liq","users_minus_points","points_minus_users","efficiency"]
_IW=[-0.10,0.10,0.00,0.00,-0.04,-0.10,0.00,-0.06,0.00,0.00,0.08,0.06,0.00,-0.04,-0.02,0.02,0.00]

class Slrabbit99ClusterEnsembleV23a(StrategyBase):
    name="slrabbit99_cluster_ensemble_v23a"
    def on_backtest_start(self):
        self.cluster_weights={c:list(_IW) for c in range(N_CLUSTERS)}
        self._sq={c:[0.01]*len(FEATURES) for c in range(N_CLUSTERS)}
        self._cluster_n={c:0.0 for c in range(N_CLUSTERS)}
    def _lr(self,c):
        return BASE_LR*(73.0/max(5.0,self._cluster_n[c]))**0.5
    @staticmethod
    def _sm(x):
        if not x: return []
        p=max(x); e=[math.exp(max(-20.,min(20.,v-p))) for v in x]; d=sum(e)or 1.; return[v/d for v in e]
    def _f(self, e, o, s):
        f=build_features(o,s.prior_win_rate(e.streamer,o.odds),event_outcomes=e.outcomes)
        return dict(bias=1.,log_odds=f["log_odds"],users_pct=o.users_pct/100.,points_pct=o.points_pct/100.,ratio=f["ratio"],rel_ratio=f["rel_ratio"],rel_odds=f["rel_odds"],rel_surprise=f["rel_surprise"],log_liq=f["log_liq"],p=f["p"],ev=f["ev"],conf=f["conf"],momentum=f["momentum"],crowd_liq=f["crowd_liq"]/10.,users_minus_points=(o.users_pct-o.points_pct)/100.,points_minus_users=(o.points_pct-o.users_pct)/100.,efficiency=f["efficiency"])
    @staticmethod
    def _sc(w,f): return sum(w[i]*f[n] for i,n in enumerate(FEATURES))
    def _ens(self,e,s):
        ci=_keyword_cluster(e.main_title)
        ff=[self._f(e,o,s) for o in e.outcomes]
        sc=[0.]*len(e.outcomes)
        for c in range(N_CLUSTERS):
            w=self.cluster_weights[c]; si=.6 if c==ci else .4/(N_CLUSTERS-1)
            for j in range(len(e.outcomes)): sc[j]+=si*self._sc(w,ff[j])
        return ff,sc,self._sm(sc),ci
    def pick(self,e,s):
        if e.streamer!="slrabbit99":return None
        _,_,p,_=self._ens(e,s)
        b,be=None,-10**9
        for o,pp in zip(e.outcomes,p):
            if(ee:=pp*o.odds-1.)>be:be,b=ee,o
        return b
    def size(self,e,o,s):
        if e.streamer!="slrabbit99"or o is None:return 0.
        _,sc,p,_=self._ens(e,s)
        si=next((i for i,oo in enumerate(e.outcomes) if oo.outcome_id==o.outcome_id),None)
        if si is None:return 0.
        edge=p[si]*o.odds-1.
        # Kelly: bet = balance * edge / (odds-1), clamped
        kelly=edge/(o.odds-1.+1e-10)
        amount=20000+300000*max(0.,kelly)  # 3x Kelly fraction
        return clamp(amount,20000.,120000.)
    def on_event_end(self,e,o,w,pn,s):
        if e.streamer!="slrabbit99":return
        ci=_keyword_cluster(e.main_title);self._cluster_n[ci]+=1.
        ff,_,p,_=self._ens(e,s)
        y=[1. if oo.outcome_id==e.winner_outcome_id else 0. for oo in e.outcomes]
        g=[pp-yy for pp,yy in zip(p,y)]
        ww=self.cluster_weights[ci];sq=self._sq[ci];lr=self._lr(ci)
        for i,n in enumerate(FEATURES):
            gg=sum(g[j]*ff[j][n] for j in range(len(e.outcomes)))
            sq[i]=RMS_DECAY*sq[i]+(1-RMS_DECAY)*gg*gg
            ww[i]-=lr*gg/(math.sqrt(sq[i])+1e-6)
