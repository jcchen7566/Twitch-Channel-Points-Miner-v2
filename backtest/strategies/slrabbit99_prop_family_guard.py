import math
from collections import defaultdict
from datetime import datetime

from ..core.features import build_features, clamp, odds_regime
from ..core.strategy_base import StrategyBase
from ..core.title_clustering import TitleClusterer


def _prop_family(title: str) -> str:
    t = (title or "").lower()
    if "失多少分" in t or "er" in t:
        return "ER_PROP"
    if "安打" in t:
        return "HITS_PROP"
    if "全壘打" in t:
        return "HR_PROP"
    if "勝投" in t:
        return "WIN_PITCH"
    if "誰會贏" in t or "誰贏" in t or "哪邊會贏" in t or "會贏嗎" in t:
        return "WHO_WIN"
    if "晉級" in t:
        return "ADVANCE"
    if "過關" in t or "不刷" in t or "最穩" in t or "命一關" in t or "關" in t:
        return "CHALLENGE"
    if "觸身" in t:
        return "HBP"
    return "OTHER"


class _Slrabbit99PropFamilyGuardMixin:

    def on_backtest_start(self):
        self.regime_edge = {k: 0.0 for k in ("A_1p8", "B_2p2", "C_3p0", "D_5p0", "E_gt5")}
        self.family_edge = defaultdict(float)

    def _score(self, event, outcome, state):
        feat = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        rg = odds_regime(outcome.odds)
        crowd_edge = 1.0 - min(2.0, feat["ratio"])
        score = feat["ev"] + 0.16 * crowd_edge + 0.14 * self.regime_edge[rg]

        family = _prop_family(event.main_title)
        odds = float(outcome.odds)
        if family == "CHALLENGE":
            score += 0.04
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                score += 0.05
            if odds > 5.0:
                score -= 0.05
        elif family == "ER_PROP":
            if odds > 3.0:
                score -= 0.12
            if odds > 4.5:
                score -= 0.12
        elif family == "HITS_PROP":
            if odds > 2.5:
                score -= 0.22
            if odds > 5.0:
                score -= 0.18
        else:
            if odds > 3.0:
                score -= 0.03
        return score, feat, family, rg

    def _base_size(self, feat, rg, outcome, family):
        amount = 80000 + 50000 * feat["ev"] + 40000 * self.regime_edge[rg]
        odds = float(outcome.odds)
        ratio = float(feat["ratio"])

        if family == "CHALLENGE":
            pass
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                amount *= 1.08
            if ratio > 1.0:
                amount *= 0.88
        elif family == "ER_PROP":
            if odds > 3.0:
                amount *= 0.55
            if odds > 5.0:
                amount *= 0.35
        elif family == "HITS_PROP":
            if odds > 2.5:
                amount *= 0.40
            if odds > 5.0:
                amount *= 0.25
        else:
            if odds > 3.0:
                amount *= 0.80

        if ratio > 1.0 and family in {"ER_PROP", "HITS_PROP"}:
            amount *= 0.85
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        rg = odds_regime(selected.odds)
        reward = 1.0 if won else -1.0
        self.regime_edge[rg] = 0.9 * self.regime_edge[rg] + 0.1 * reward
        family = _prop_family(event.main_title)
        self.family_edge[family] = 0.9 * self.family_edge.get(family, 0.0) + 0.1 * reward


class Slrabbit99PropFamilyGuardSizerV1(_Slrabbit99PropFamilyGuardMixin, StrategyBase):
    name = "slrabbit99_prop_family_guard_sizer_v1"

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None

        best = None
        best_score = -10**9
        for o in event.outcomes:
            score, feat, family, rg = self._score(event, o, state)
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0
        score, feat, family, rg = self._score(event, selected, state)
        amount = self._base_size(feat, rg, selected, family)
        return clamp(amount, 20000, 120000)

class Slrabbit99PropFamilyGuardSizerV2(Slrabbit99PropFamilyGuardSizerV1):
    name = "slrabbit99_prop_family_guard_sizer_v2"

    def _score(self, event, outcome, state):
        feat = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        rg = odds_regime(outcome.odds)
        crowd_edge = 1.0 - min(2.0, feat["ratio"])
        score = feat["ev"] + 0.24 * crowd_edge + 0.16 * self.regime_edge[rg]

        family = _prop_family(event.main_title)
        odds = float(outcome.odds)
        if family == "CHALLENGE":
            score += 0.04
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                score += 0.05
            if odds > 5.0:
                score -= 0.05
        elif family == "ER_PROP":
            if odds > 3.0:
                score -= 0.12
            if odds > 4.5:
                score -= 0.12
        elif family == "HITS_PROP":
            if odds > 2.5:
                score -= 0.22
            if odds > 5.0:
                score -= 0.18
        else:
            if odds > 3.0:
                score -= 0.03
        return score, feat, family, rg

    def _base_size(self, feat, rg, outcome, family):
        amount = 50000 + 40000 * feat["ev"] + 10000 * self.regime_edge[rg]
        odds = float(outcome.odds)
        ratio = float(feat["ratio"])

        if family == "CHALLENGE":
            pass
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                amount *= 1.12
            if ratio > 1.0:
                amount *= 0.84
        elif family == "ER_PROP":
            if odds > 3.0:
                amount *= 0.40
            if odds > 5.0:
                amount *= 0.25
        elif family == "HITS_PROP":
            if odds > 2.5:
                amount *= 0.25
            if odds > 5.0:
                amount *= 0.10
        else:
            if odds > 3.0:
                amount *= 0.60

        if ratio > 1.0 and family in {"ER_PROP", "HITS_PROP"}:
            amount *= 0.75
        return amount


# ── v10: Adaptive regime-edge decay ──────────────────────────────────
class Slrabbit99PropFamilyGuardSizerV10(Slrabbit99PropFamilyGuardSizerV2):
    """Monitors trailing odds for game-type shifts.  When detected,
    regime_edge decay speeds from 0.9 → 0.5 for faster adaptation."""
    name = "slrabbit99_prop_family_guard_sizer_v10"

    def on_backtest_start(self):
        super().on_backtest_start()
        self._odds_trail: list[float] = []

    def on_event_end(self, event, selected, won, pnl, state):
        odds = float(selected.odds)
        self._odds_trail.append(odds)
        if len(self._odds_trail) > 30:
            self._odds_trail.pop(0)

        if len(self._odds_trail) >= 12:
            recent = self._odds_trail[-5:]
            baseline = self._odds_trail[:-5]
            shift = (sum(recent)/5) > 2.0 * (sum(baseline)/max(1,len(baseline))) or \
                    (sum(recent)/5) < 0.5 * (sum(baseline)/max(1,len(baseline)))
            decay = 0.50 if shift else 0.90
        else:
            decay = 0.90

        rg = odds_regime(odds)
        reward = 1.0 if won else -1.0
        self.regime_edge[rg] = decay * self.regime_edge[rg] + (1.0 - decay) * reward
        family = _prop_family(event.main_title)
        self.family_edge[family] = 0.9 * self.family_edge.get(family, 0.0) + 0.1 * reward


# ── v4 strategies ────────────────────────────────────────────────────


def _pp(event, outcome):
    """Safely extract points_pct from outcome."""
    return float(getattr(outcome, "points_pct", 0.0) or 0.0)


def _is_fav(outcome, outcomes):
    min_o = min(float(o.odds) for o in outcomes)
    return float(outcome.odds) == min_o


class Slrabbit99PropFamilyGuardSizerV4a(_Slrabbit99PropFamilyGuardMixin, StrategyBase):
    """pp_alpha: points_pct as primary scoring factor.

    Core discovery: pp (points_pct) has +0.37 correlation with winning
    for slrabbit99, stronger than any other single feature. This strategy
    uses pp as the primary signal modulator.
    """
    name = "slrabbit99_prop_family_guard_sizer_v4a"

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        best = None
        best_score = -10**9
        for o in event.outcomes:
            score, feat, _, _ = self.score_v4a(event, o, state)
            if score > best_score:
                best_score = score
                best = o
        return best

    def score_v4a(self, event, outcome, state):
        feat = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        rg = odds_regime(outcome.odds)
        family = _prop_family(event.main_title)
        odds = float(outcome.odds)
        pp = _pp(event, outcome)
        ratio = float(feat["ratio"])

        # Base score: ev modulated by pp/50 (pp>50 boosts, pp<50 penalizes)
        score = feat["ev"] * (pp / 50.0)

        crowd_edge = 1.0 - min(2.0, ratio)
        score += 0.20 * crowd_edge

        if 0.9 < ratio < 1.0:
            score += 0.08
        if ratio < 0.7:
            score += 0.06

        if family == "CHALLENGE":
            score += 0.06
            if pp >= 50:
                score += 0.10

        score += 0.14 * self.regime_edge[rg]

        if ratio >= 1.0:
            score -= 0.06

        return score, feat, family, rg

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0
        _, feat, family, rg = self.score_v4a(event, selected, state)
        odds = float(selected.odds)
        ratio = float(feat["ratio"])
        pp = _pp(event, selected)

        amount = 40000 + 30000 * feat["ev"] + 15000 * self.regime_edge[rg]

        if pp >= 70:
            amount *= 1.40
        elif pp >= 60:
            amount *= 1.25
        elif pp >= 50:
            amount *= 1.10

        if 0.9 < ratio < 1.0:
            amount *= 1.35
        elif ratio < 0.7:
            amount *= 1.20

        if family == "CHALLENGE" and pp >= 50:
            amount *= 1.20

        if ratio >= 1.0:
            amount *= 0.80
            if ratio >= 1.2:
                amount *= 0.75

        if family == "ER_PROP" and odds > 3.0:
            amount *= 0.45
        if family == "HITS_PROP" and odds > 2.5:
            amount *= 0.30

        if _is_sunday(event.timestamp):
            amount *= _SUNDAY_CUT
        return clamp(amount, 20000, 120000)


class Slrabbit99PropFamilyGuardSizerV4b(_Slrabbit99PropFamilyGuardMixin, StrategyBase):
    """whale_master: ratio-based primary strategy.

    LightWhale (ratio 0.9-1.0) → 75% WR at avg_odds=2.44.
    Heavy Whale (ratio<0.7) → 72.5% WR.
    Strategy organizes around ratio zones.
    """
    name = "slrabbit99_prop_family_guard_sizer_v4b"

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        best = None
        best_score = -10**9
        for o in event.outcomes:
            score, feat, _, _ = self.score_v4b(event, o, state)
            if score > best_score:
                best_score = score
                best = o
        return best

    def score_v4b(self, event, outcome, state):
        feat = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        rg = odds_regime(outcome.odds)
        family = _prop_family(event.main_title)
        odds = float(outcome.odds)
        pp = _pp(event, outcome)
        ratio = float(feat["ratio"])

        score = feat["ev"]

        if 0.9 < ratio < 1.0:
            score += 0.18
            if pp >= 60:
                score += 0.10
        elif ratio < 0.7:
            score += 0.14
        elif ratio < 0.9:
            score += 0.06
        elif 1.0 <= ratio < 1.2:
            score -= 0.04
        else:
            score -= 0.10

        if _is_fav(outcome, event.outcomes):
            score += 0.06
            if 0.9 < ratio < 1.0:
                score += 0.08

        if family == "CHALLENGE":
            score += 0.05
            if pp >= 50:
                score += 0.08

        score += 0.14 * self.regime_edge[rg]
        return score, feat, family, rg

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0
        _, feat, family, rg = self.score_v4b(event, selected, state)
        odds = float(selected.odds)
        ratio = float(feat["ratio"])
        pp = _pp(event, selected)

        amount = 40000 + 30000 * feat["ev"] + 15000 * self.regime_edge[rg]

        if 0.9 < ratio < 1.0:
            amount *= 1.50
            if pp >= 60:
                amount *= 1.25
        elif ratio < 0.7:
            amount *= 1.30
        elif ratio < 0.9:
            amount *= 1.10
        elif 1.0 <= ratio < 1.2:
            amount *= 0.70
        else:
            amount *= 0.40

        if _is_fav(selected, event.outcomes):
            amount *= 1.15

        if family == "CHALLENGE" and pp >= 50:
            amount *= 1.20

        if _is_sunday(event.timestamp):
            amount *= _SUNDAY_CUT
        return clamp(amount, 20000, 120000)


class Slrabbit99PropFamilyGuardSizerV4c(_Slrabbit99PropFamilyGuardMixin, StrategyBase):
    """two_factor: minimal model — only pp + ratio.

    No title families, no regime edges, no prop type bonuses.
    Tests whether the simplest signals are sufficient.
    """
    name = "slrabbit99_prop_family_guard_sizer_v4c"

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        best = None
        best_score = -10**9
        for o in event.outcomes:
            score, feat = self.score_v4c(event, o, state)
            if score > best_score:
                best_score = score
                best = o
        return best

    def score_v4c(self, event, outcome, state):
        feat = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        pp = float(outcome.points_pct)
        ratio = float(feat["ratio"])

        pp_factor = (pp - 50.0) / 100.0
        ratio_inv = (1.0 - min(2.0, ratio)) * 0.20
        score = feat["ev"] + pp_factor + ratio_inv + 0.02
        return score, feat

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0
        score, feat = self.score_v4c(event, selected, state)
        pp = _pp(event, selected)
        ratio = float(feat["ratio"])

        amount = 40000 + 40000 * feat["ev"]

        if pp >= 60:
            amount *= 1.30
        if pp >= 75:
            amount *= 1.30

        if 0.9 < ratio < 1.0:
            amount *= 1.40
        elif ratio < 0.9:
            amount *= 1.15
        elif ratio >= 1.0:
            amount *= 0.65
            if ratio >= 1.5:
                amount *= 0.50

        if _is_sunday(event.timestamp):
            amount *= _SUNDAY_CUT
        return clamp(amount, 20000, 120000)


class Slrabbit99PropFamilyGuardSizerV4d(_Slrabbit99PropFamilyGuardMixin, StrategyBase):
    """hybrid: v3 scoring + v4b ratio-zone sizing.

    v3 has the best scoring function (proven on full history).
    v4b has the best sizing logic (ratio-zone multipliers).
    This combines both: best score for picking, conservative sizing for risk.
    """
    name = "slrabbit99_prop_family_guard_sizer_v4d"

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None
        best = None
        best_score = -10**9
        for o in event.outcomes:
            score, _, _, _ = self.score_v4d(event, o, state)
            if score > best_score:
                best_score = score
                best = o
        return best

    def score_v4d(self, event, outcome, state):
        """Identical to v3 scoring."""
        feat = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        rg = odds_regime(outcome.odds)
        family = _prop_family(event.main_title)
        odds = float(outcome.odds)
        crowd_edge = 1.0 - min(2.0, feat["ratio"])
        score = feat["ev"] + 0.24 * crowd_edge + 0.16 * self.regime_edge[rg]
        if family == "CHALLENGE":
            score += 0.04
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                score += 0.05
            if odds > 5.0:
                score -= 0.05
        elif family == "ER_PROP":
            if odds > 3.0:
                score -= 0.12
            if odds > 4.5:
                score -= 0.12
        elif family == "HITS_PROP":
            if odds > 2.5:
                score -= 0.22
            if odds > 5.0:
                score -= 0.18
        else:
            if odds > 3.0:
                score -= 0.03
        return score, feat, family, rg

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0
        _, feat, family, rg = self.score_v4d(event, selected, state)
        odds = float(selected.odds)
        ratio = float(feat["ratio"])
        pp = _pp(event, selected)

        # Same base as v3
        amount = 50000 + 40000 * feat["ev"] + 10000 * self.regime_edge[rg]

        # v4b-style ratio-zone sizing
        if 0.9 < ratio < 1.0:
            amount *= 1.40
            if pp >= 60:
                amount *= 1.20
        elif ratio < 0.7:
            amount *= 1.20
        elif ratio < 0.9:
            amount *= 1.10
        elif 1.0 <= ratio < 1.2:
            amount *= 0.75
        else:
            amount *= 0.50

        if _is_fav(selected, event.outcomes):
            amount *= 1.10

        if family == "CHALLENGE" and pp >= 50:
            amount *= 1.15

        if family == "ER_PROP" and odds > 3.0:
            amount *= 0.40
        if family == "HITS_PROP" and odds > 2.5:
            amount *= 0.25
        if family == "OTHER" and odds > 3.0:
            amount *= 0.60

        if _is_sunday(event.timestamp):
            amount *= _SUNDAY_CUT
        return clamp(amount, 20000, 120000)


class Slrabbit99PropFamilyGuardSizerV5(Slrabbit99PropFamilyGuardSizerV2):
    """Cluster-adaptor: auto-detects game types via title clustering.

    During preload_history:
      1. Fits TitleClusterer (char-bigram TF-IDF → DBSCAN) on all unique titles.
      2. Computes per-cluster average |1 - ratio| deviation as a measure of
         market efficiency (deviation near 0 = efficient = low edge).
      3. Each cluster gets a sizing multiplier — efficient clusters bet smaller.

    Scoring and sizing otherwise identical to v2 — this is a minimally invasive
    wrapper that lets clusters self-correct without manual game lists.
    """
    name = "slrabbit99_prop_family_guard_sizer_v5"

    def __init__(self):
        super().__init__()
        self.clusterer = TitleClusterer()
        self._size_mult = {}
        self._score_bonus = {}
        self._trained = False

    def on_backtest_start(self):
        super().on_backtest_start()
        self._size_mult = {}
        self._score_bonus = {}
        self._trained = False

    def preload_history(self, events: list) -> None:
        streamer_events = [e for e in events if e.streamer == "slrabbit99"]
        titles = sorted(set(e.main_title for e in streamer_events))
        if len(titles) < 3:
            return

        self.clusterer.fit(titles)

        cluster_wr: dict[int, dict] = defaultdict(
            lambda: {"n": 0, "w": 0}
        )
        for ev in streamer_events:
            if not ev.label_known:
                continue
            cid = self.clusterer.predict(ev.main_title)

            best = None
            best_scr = -999.0
            for o in ev.outcomes:
                hw, ht = float(o.hw), float(o.ht)
                prior = hw / max(1.0, ht) if ht > 3 else 0.5
                ev_val = (float(o.odds) - 1.0) * prior - (1.0 - prior)

                ratio = float(o.users_pct) / max(1.0, float(o.points_pct))
                crowd_edge = 1.0 - min(2.0, ratio)
                scr = ev_val + 0.24 * crowd_edge
                if scr > best_scr:
                    best_scr = scr
                    best = o

            if best:
                cluster_wr[cid]["n"] += 1
                if best.outcome_id == ev.winner_outcome_id:
                    cluster_wr[cid]["w"] += 1

        total_n = sum(d["n"] for d in cluster_wr.values())
        total_w = sum(d["w"] for d in cluster_wr.values())
        global_wr = (total_w + 1.0) / (total_n + 2.0) if total_n > 0 else 0.5

        for cid, d in cluster_wr.items():
            n = d["n"]
            if n < 4:
                self._size_mult[cid] = 1.0
                self._score_bonus[cid] = 0.0
                continue

            wr = d["w"] / n
            lam = n / (n + 10.0)
            shrunk = lam * wr + (1.0 - lam) * global_wr

            mult = shrunk / global_wr
            self._size_mult[cid] = clamp(mult, 0.30, 1.50)

            bonus = (shrunk - global_wr) * 0.20
            self._score_bonus[cid] = clamp(bonus, -0.15, 0.15)

        self._trained = True

    def _get_cid(self, title: str) -> int:
        return self.clusterer.predict(title) if self._trained else -1

    def _score(self, event, outcome, state):
        score, feat, family, rg = super()._score(event, outcome, state)
        if self._trained:
            cid = self._get_cid(event.main_title)
            score += self._score_bonus.get(cid, 0.0)
        return score, feat, family, rg

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0
        score, feat, family, rg = self._score(event, selected, state)
        amount = self._base_size(feat, rg, selected, family)
        if self._trained:
            cid = self._get_cid(event.main_title)
            amount *= self._size_mult.get(cid, 1.0)
        return clamp(amount, 20000, 120000)


# ── v8: Per-family crowd_edge weights ──────────────────────────────────
# Analysis shows each family has a different optimal crowd_edge weight:
#   WHO_WIN:   0.04  (whales have no info edge for who-wins)
#   ER_PROP:  -0.20  (bet against whale signal on ER props)
#   HITS_PROP:-0.20  (bet against whale signal on hits props)
#   HR_PROP:  -0.20  (small sample, but consistent with ER/HITS)
#   CHALLENGE: 0.24  (same as v2 global optimum)
#   default:   0.24  (unchanged)
#
# v8 overrides _score to use per-family crowd_edge, leaving regime_edge
# and family bonuses identical to v2.  Everything else inherits from v2.
_FAMILY_CROWD_WEIGHT = {
    "WHO_WIN": 0.04,
    "ER_PROP": -0.20,
    "HITS_PROP": -0.20,
    "HR_PROP": -0.20,
}


class Slrabbit99PropFamilyGuardSizerV8(Slrabbit99PropFamilyGuardSizerV2):
    name = "slrabbit99_prop_family_guard_sizer_v8"

    def _score(self, event, outcome, state):
        feat = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        rg = odds_regime(outcome.odds)
        crowd_edge = 1.0 - min(2.0, feat["ratio"])
        score = feat["ev"] + _FAMILY_CROWD_WEIGHT.get(_prop_family(event.main_title), 0.24) * crowd_edge
        score += 0.16 * self.regime_edge[rg]

        family = _prop_family(event.main_title)
        odds = float(outcome.odds)
        if family == "CHALLENGE":
            score += 0.04
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                score += 0.05
            if odds > 5.0:
                score -= 0.05
        elif family == "ER_PROP":
            if odds > 3.0:
                score -= 0.12
            if odds > 4.5:
                score -= 0.12
        elif family == "HITS_PROP":
            if odds > 2.5:
                score -= 0.22
            if odds > 5.0:
                score -= 0.18
        else:
            if odds > 3.0:
                score -= 0.03
        return score, feat, family, rg


# ── v9: Behavioral-feature regime clustering via decision tree ─────────
# Instead of clustering title text, cluster events by observable market
# features (avg_ratio, avg_odds, liquidity, DOW).  A shallow decision tree
# finds the splits that best separate winning vs losing events.
#
# Each leaf = a behavioral regime with its own crowd_edge weight and
# sizing multiplier.  This lets the strategy adapt to market structure
# shifts (e.g. compressed odds + convergent ratio → efficient market →
# bet smaller) without manual regime definitions.
from sklearn.tree import DecisionTreeClassifier


def _event_regime_features(event):
    """Event-level features observable before picking (outcome-independent)."""
    ratios = []
    log_odds = []
    total_points = 0.0
    for o in event.outcomes:
        ratios.append(float(o.users_pct) / max(1.0, float(o.points_pct)))
        log_odds.append(math.log(max(1.01, float(o.odds))))
        total_points = max(total_points, float(o.total_points))

    avg_ratio = sum(ratios) / len(ratios)
    avg_log_odds = sum(log_odds) / len(log_odds)
    log_liq = math.log10(max(1.0, total_points))

    try:
        dt = datetime.strptime(event.timestamp, "%Y-%m-%d %H:%M:%S")
        dow = dt.weekday()
    except Exception:
        dow = 0
    dow_sin = math.sin(2 * math.pi * dow / 7)
    dow_cos = math.cos(2 * math.pi * dow / 7)

    return [avg_ratio, avg_log_odds, log_liq, dow_sin, dow_cos]


class Slrabbit99PropFamilyGuardSizerV9(Slrabbit99PropFamilyGuardSizerV2):
    name = "slrabbit99_prop_family_guard_sizer_v9"

    def __init__(self):
        super().__init__()
        self._tree = None
        self._leaf_params = {}
        self._trained = False

    def on_backtest_start(self):
        super().on_backtest_start()
        self._trained = False
        self._leaf_params = {}

    def preload_history(self, events: list) -> None:
        pool = [e for e in events if e.streamer == "slrabbit99" and e.label_known]
        if len(pool) < 30:
            return

        X = []
        y = []
        for ev in pool:
            X.append(_event_regime_features(ev))
            won = 0
            if ev.selected_outcome_id and ev.winner_outcome_id:
                for o in ev.outcomes:
                    if o.outcome_id == ev.selected_outcome_id:
                        if o.outcome_id == ev.winner_outcome_id:
                            won = 1
                        break
            y.append(won)

        tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=15, random_state=42)
        tree.fit(X, y)
        self._tree = tree

        leaf_ids = tree.apply(X)
        unique_leaves = sorted(set(leaf_ids))
        global_wr = sum(y) / max(1, len(y))

        for lid in unique_leaves:
            idxs = [i for i, l in enumerate(leaf_ids) if l == lid]
            n = len(idxs)
            w = sum(y[i] for i in idxs)
            wr = w / n if n > 0 else 0.5

            # Shrink toward global
            lam = n / (n + 12.0)
            shrunk = lam * wr + (1.0 - lam) * global_wr

            best_w = 0.24
            best_roi = -1e9
            for cw in [x * 0.08 for x in range(-5, 9)]:
                leaf_bet = 0.0
                leaf_gain = 0.0
                for i in idxs:
                    ev = pool[i]
                    best_score = -999.0
                    best_o = None
                    for o in ev.outcomes:
                        oo = float(o.odds)
                        up = float(o.users_pct)
                        pp = float(o.points_pct)
                        ratio = up / max(1.0, pp)
                        edge = 1.0 - min(2.0, ratio)
                        hw = float(o.hw)
                        ht = float(o.ht)
                        prior = hw / max(1.0, ht) if ht > 3 else 0.5
                        p = 0.50 * prior + 0.50 * (1.0 / oo)
                        ev_val = p * oo - 1.0
                        score = ev_val + cw * edge
                        if score > best_score:
                            best_score = score
                            best_o = o
                    if best_o:
                        amt = 50000 + 40000 * (best_score - cw * (1.0 - min(2.0, float(best_o.users_pct) / max(1.0, float(best_o.points_pct)))))
                        amt = max(20000, min(120000, amt))
                        leaf_bet += amt
                        if best_o.outcome_id == ev.winner_outcome_id:
                            leaf_gain += amt * (float(best_o.odds) - 1.0)
                        else:
                            leaf_gain -= amt
                roi = (leaf_gain / leaf_bet * 100) if leaf_bet > 0 else -100
                if roi > best_roi:
                    best_roi = roi
                    best_w = cw

            # Sizing multiplier: shrunk WR relative to global
            size_mult = shrunk / max(0.01, global_wr)  # e.g. 0.60/0.50 = 1.2
            size_mult = max(0.50, min(1.50, size_mult))

            self._leaf_params[lid] = {
                "crowd_w": best_w,
                "size_mult": size_mult,
                "shrunken_wr": shrunk,
            }

        self._trained = True

    def _score(self, event, outcome, state):
        feat = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        rg = odds_regime(outcome.odds)
        crowd_edge = 1.0 - min(2.0, feat["ratio"])

        w_crowd = 0.24
        if self._trained and self._tree is not None:
            x = [_event_regime_features(event)]
            lid = int(self._tree.apply(x)[0])
            w_crowd = self._leaf_params.get(lid, {}).get("crowd_w", 0.24)

        score = feat["ev"] + w_crowd * crowd_edge + 0.16 * self.regime_edge[rg]

        family = _prop_family(event.main_title)
        odds = float(outcome.odds)
        if family == "CHALLENGE":
            score += 0.04
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                score += 0.05
            if odds > 5.0:
                score -= 0.05
        elif family == "ER_PROP":
            if odds > 3.0:
                score -= 0.12
            if odds > 4.5:
                score -= 0.12
        elif family == "HITS_PROP":
            if odds > 2.5:
                score -= 0.22
            if odds > 5.0:
                score -= 0.18
        else:
            if odds > 3.0:
                score -= 0.03
        return score, feat, family, rg

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0
        score, feat, family, rg = self._score(event, selected, state)
        amount = self._base_size(feat, rg, selected, family)

        if self._trained and self._tree is not None:
            x = [_event_regime_features(event)]
            lid = int(self._tree.apply(x)[0])
            mult = self._leaf_params.get(lid, {}).get("size_mult", 1.0)
            amount *= mult

        return clamp(amount, 20000, 120000)


class Slrabbit99PropFamilyGuardSizerV6(Slrabbit99PropFamilyGuardSizerV2):
    """v2 + per-family momentum tracking.

    Each prop family (WHO_WIN, ER_PROP, CHALLENGE, etc.) gets its own
    exponentially-weighted momentum tracker via on_event_end.  The momentum
    feeds back into both scoring and sizing so that families on a cold streak
    naturally bet smaller until they regress.

    This lets the strategy adapt to game-type shifts automatically: when the
    streamer switches to a different game format, the new family's momentum
    builds from history and self-corrects without any manual config change.
    """
    name = "slrabbit99_prop_family_guard_sizer_v6"

    def _score(self, event, outcome, state):
        score, feat, family, rg = super()._score(event, outcome, state)
        score += 0.08 * self.family_edge.get(family, 0.0)
        return score, feat, family, rg

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0
        score, feat, family, rg = self._score(event, selected, state)
        amount = self._base_size(feat, rg, selected, family)

        family_momentum = self.family_edge.get(family, 0.0)
        amount *= (1.0 + 0.15 * family_momentum)

        return clamp(amount, 20000, 120000)


_LIGHTWHALE_SIZING = 1.30
_SUNDAY_CUT = 0.75  # Sunday → -25% sizing


def _is_sunday(ts: str) -> bool:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").weekday() == 6
    except Exception:
        return False


class Slrabbit99PropFamilyGuardSizerV3(_Slrabbit99PropFamilyGuardMixin, StrategyBase):
    name = "slrabbit99_prop_family_guard_sizer_v3"

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None

        best = None
        best_score = -10**9
        for o in event.outcomes:
            score, feat, family, rg = self.score_v2(event, o, state)
            if score > best_score:
                best_score = score
                best = o
        return best

    def score_v2(self, event, outcome, state):
        feat = build_features(outcome, state.prior_win_rate(event.streamer, outcome.odds), event_outcomes=event.outcomes)
        rg = odds_regime(outcome.odds)
        family = _prop_family(event.main_title)
        odds = float(outcome.odds)
        crowd_edge = 1.0 - min(2.0, feat["ratio"])
        score = feat["ev"] + 0.24 * crowd_edge + 0.16 * self.regime_edge[rg]
        if family == "CHALLENGE":
            score += 0.04
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                score += 0.05
            if odds > 5.0:
                score -= 0.05
        elif family == "ER_PROP":
            if odds > 3.0:
                score -= 0.12
            if odds > 4.5:
                score -= 0.12
        elif family == "HITS_PROP":
            if odds > 2.5:
                score -= 0.22
            if odds > 5.0:
                score -= 0.18
        else:
            if odds > 3.0:
                score -= 0.03
        return score, feat, family, rg

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0
        _, feat, family, rg = self.score_v2(event, selected, state)
        amount = self.base_size_v2(feat, rg, selected, family)

        if _is_sunday(event.timestamp):
            amount *= _SUNDAY_CUT

        if 0.9 < feat["ratio"] < 1.0:
            amount *= _LIGHTWHALE_SIZING

        return clamp(amount, 20000, 120000)

    def base_size_v2(self, feat, rg, outcome, family):
        amount = 50000 + 40000 * feat["ev"] + 10000 * self.regime_edge[rg]
        odds = float(outcome.odds)
        ratio = float(feat["ratio"])

        if family == "CHALLENGE":
            pass
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                amount *= 1.12
            if ratio > 1.0:
                amount *= 0.84
        elif family == "ER_PROP":
            if odds > 3.0:
                amount *= 0.40
            if odds > 5.0:
                amount *= 0.25
        elif family == "HITS_PROP":
            if odds > 2.5:
                amount *= 0.25
            if odds > 5.0:
                amount *= 0.10
        else:
            if odds > 3.0:
                amount *= 0.60

        if ratio > 1.0 and family in {"ER_PROP", "HITS_PROP"}:
            amount *= 0.75
        return amount

