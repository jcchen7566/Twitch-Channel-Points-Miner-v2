"""v23b — v9 (dynamic clustering + similarity-weighted ensemble) + v22d (RMSprop + cluster LR)."""
import math
from collections import defaultdict
from ..core.strategy_base import StrategyBase
from ..core.features import build_features, clamp

def _keyword_cluster(t: str) -> int | None:
    t=(t or"").strip().lower()
    if any(k in t for k in["誰會贏","哪邊會贏"," vs ","誰贏"]):return 0
    if any(k in t for k in["失分","勝投","全壘打","安打","觸身","er ","er)"]):return 1
    if any(k in t for k in["命一關不刷","一關不刷"]):return 2
    if any(k in t for k in["第幾名","開到幾星","抽"]):return 3
    if any(k in t for k in["會不會過關","能不能過關","一把決勝負"]):return 4
    if"最穩的那台"in t:return 5
    if"丟骨頭"in t:return 6
    return None  # CATCH_ALL → let DCM create dynamic cluster

class _C:
    __slots__=("ns","c")
    def __init__(s):s.ns=defaultdict(float);s.c=0
    def add(s,v):
        for k,vv in v.items():s.ns[k]+=vv
        s.c+=1
    @property
    def cent(s):
        return{k:v/s.c for k,v in s.ns.items()}if s.c else{}
    def cos(s,v):
        c=s.cent
        if not c or not v:return 0.
        d=sum(vv*c.get(k,0.) for k,vv in v.items())
        n1=math.sqrt(sum(vv*vv for vv in v.values()))or 1.;n2=math.sqrt(sum(vv*vv for vv in c.values()))or 1.
        return d/(n1*n2)

class _DCM:
    def __init__(s,th=.25):
        s.th=th;s.cs=[];s.tm={}
        for ex in[["BABA VS HENDERSON 誰贏"],["大谷今天會失分嗎"],["命一關不刷 這關會過嗎"],["第幾名 改車 來賓開到幾星"],["會不會過關"],["最穩的那台"],["丟骨頭"]]:
            c=_C()
            for e in ex:
                v=s._ng(e)
                if any(x>0 for x in v.values()):c.add(v)
            s.cs.append(c)
    @staticmethod
    def _ng(t):
        t=t.lower().strip();g=defaultdict(float)
        for n in(2,3):
            for i in range(len(t)-n+1):g[t[i:i+n]]+=1.
        return dict(g)
    def assign(s,title):
        r=s.tm.get(title)
        if r is not None:return r
        v=s._ng(title)
        bi,bs=-1,-1.
        for i,c in enumerate(s.cs):
            si=c.cos(v)
            if si>bs:bs,bi=si,i
        if bs<s.th:
            kw=_keyword_cluster(title)
            if kw is not None:s.cs[kw].add(v);s.tm[title]=kw;return kw
        if bs>=s.th:s.cs[bi].add(v);s.tm[title]=bi;return bi
        ni=len(s.cs);c=_C();c.add(v);s.cs.append(c);s.tm[title]=ni;return ni

N_CLUSTERS=8;BASE_LR=0.09;RMS_DECAY=0.85
FEATURES=["bias","log_odds","users_pct","points_pct","ratio","rel_ratio","rel_odds","rel_surprise","log_liq","p","ev","conf","momentum","crowd_liq","users_minus_points","points_minus_users","efficiency"]
_IW=[-0.10,0.10,0.00,0.00,-0.04,-0.10,0.00,-0.06,0.00,0.00,0.08,0.06,0.00,-0.04,-0.02,0.02,0.00]

class Slrabbit99ClusterEnsembleV23b(StrategyBase):
    name="slrabbit99_cluster_ensemble_v23b"
    def on_backtest_start(self):
        self.cluster_weights={c:list(_IW) for c in range(N_CLUSTERS)}
        self._sq={c:[0.01]*len(FEATURES) for c in range(N_CLUSTERS)}
        self._cluster_n={c:0.0 for c in range(N_CLUSTERS)}
        self._cm=_DCM()
    def _lr(self,c):
        return BASE_LR*(73./max(5.,self._cluster_n[c]))**0.5
    @staticmethod
    def _sm(x):
        if not x: return[]
        p=max(x);e=[math.exp(max(-20.,min(20.,v-p))) for v in x];d=sum(e)or 1.;return[v/d for v in e]
    def _f(self,e,o,s):
        f=build_features(o,s.prior_win_rate(e.streamer,o.odds),event_outcomes=e.outcomes)
        return dict(bias=1.,log_odds=f["log_odds"],users_pct=o.users_pct/100.,points_pct=o.points_pct/100.,ratio=f["ratio"],rel_ratio=f["rel_ratio"],rel_odds=f["rel_odds"],rel_surprise=f["rel_surprise"],log_liq=f["log_liq"],p=f["p"],ev=f["ev"],conf=f["conf"],momentum=f["momentum"],crowd_liq=f["crowd_liq"]/10.,users_minus_points=(o.users_pct-o.points_pct)/100.,points_minus_users=(o.points_pct-o.users_pct)/100.,efficiency=f["efficiency"])
    @staticmethod
    def _sc(w,f):return sum(w[i]*f[n] for i,n in enumerate(FEATURES))
    def _cw(self,c):
        if c not in self.cluster_weights:
            self.cluster_weights[c]=list(_IW)
            self._sq[c]=[0.01]*len(FEATURES)
        return self.cluster_weights[c]
    def _ens(self,e,s):
        title=e.main_title;ci=self._cm.assign(title)
        ff=[self._f(e,o,s) for o in e.outcomes]
        sc=[0.]*len(e.outcomes);tw=0.
        v=self._cm._ng(title)
        for c_idx in range(len(self._cm.cs)):
            w=self._cw(c_idx)
            si=max(0.01,self._cm.cs[c_idx].cos(v));tw+=si
            for j in range(len(e.outcomes)):sc[j]+=si*self._sc(w,ff[j])
        if tw>0:sc=[s/tw for s in sc]
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
        ordered=sorted(sc,reverse=True);margin=ordered[0]-ordered[1] if len(ordered)>1 else ordered[0]
        edge=p[si]*o.odds-1.
        return clamp(20000+100000*max(0.,edge)*(1./(1.+math.exp(-margin)) if margin<35 else 1.),20000.,120000.)
    def on_event_end(self,e,o,w,pn,s):
        if e.streamer!="slrabbit99":return
        ci=self._cm.assign(e.main_title);self._cluster_n.setdefault(ci,0.0);self._cluster_n[ci]+=1.
        # ensure weight vector exists for dynamic clusters
        if ci not in self.cluster_weights:
            self.cluster_weights[ci]=list(_IW)
            self._sq[ci]=[0.01]*len(FEATURES)
        ff,_,p,_=self._ens(e,s)
        y=[1. if oo.outcome_id==e.winner_outcome_id else 0. for oo in e.outcomes]
        g=[pp-yy for pp,yy in zip(p,y)]
        ww=self.cluster_weights[ci];sq=self._sq[ci];lr=self._lr(ci)
        for i,n in enumerate(FEATURES):
            gg=sum(g[j]*ff[j][n] for j in range(len(e.outcomes)))
            sq[i]=RMS_DECAY*sq[i]+(1-RMS_DECAY)*gg*gg
            ww[i]-=lr*gg/(math.sqrt(sq[i])+1e-6)
