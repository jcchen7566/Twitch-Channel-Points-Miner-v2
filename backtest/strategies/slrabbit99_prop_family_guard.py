import math
from datetime import datetime

from ..core.features import build_features, clamp, odds_regime
from ..core.strategy_base import StrategyBase


def _prop_family(title: str) -> str:
    t = (title or "").lower()
    if "失多少分" in t or "er" in t:
        return "ER_PROP"
    if "安打" in t:
        return "HITS_PROP"
    if "全壘打" in t:
        return "HR_PROP"
    if "過關" in t or "不刷" in t or "最穩" in t or "命一關" in t or "關" in t:
        return "CHALLENGE"
    return "OTHER"


class _Slrabbit99PropFamilyGuardMixin:

    def on_backtest_start(self):
        self.regime_edge = {k: 0.0 for k in ("A_1p8", "B_2p2", "C_3p0", "D_5p0", "E_gt5")}

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


_LIGHTWHALE_SIZING = 1.30  # ratio 0.9-1.0 → +30% sizing
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

