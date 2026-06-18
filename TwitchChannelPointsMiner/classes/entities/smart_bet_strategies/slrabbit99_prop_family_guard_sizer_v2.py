import sqlite3
from collections import defaultdict

from .base import SmartBetStrategyBase


class _ReplayState:
    def __init__(self):
        self.global_stats = {"n": 0.0, "w": 0.0}
        self.streamer_stats = defaultdict(lambda: {"n": 0.0, "w": 0.0})
        self.streamer_regime_stats = defaultdict(lambda: {"n": 0.0, "w": 0.0})
        self.regime_edge = {k: 0.0 for k in ("A_1p8", "B_2p2", "C_3p0", "D_5p0", "E_gt5")}

    @staticmethod
    def _regime(odds: float) -> str:
        if odds <= 1.8:
            return "A_1p8"
        if odds <= 2.2:
            return "B_2p2"
        if odds <= 3.0:
            return "C_3p0"
        if odds <= 5.0:
            return "D_5p0"
        return "E_gt5"

    def prior_win_rate(self, streamer: str, odds: float) -> float:
        regime = self._regime(odds)
        sr = self.streamer_regime_stats[(streamer, regime)]
        ss = self.streamer_stats[streamer]
        gg = self.global_stats
        if sr["n"] >= 8:
            return (sr["w"] + 1.0) / (sr["n"] + 2.0)
        if ss["n"] >= 10:
            return (ss["w"] + 1.5) / (ss["n"] + 3.0)
        if gg["n"] > 0:
            return (gg["w"] + 2.0) / (gg["n"] + 4.0)
        return 0.5

    def update(self, streamer: str, odds: float, won: bool) -> None:
        regime = self._regime(odds)
        self.global_stats["n"] += 1.0
        self.global_stats["w"] += 1.0 if won else 0.0

        s = self.streamer_stats[streamer]
        s["n"] += 1.0
        s["w"] += 1.0 if won else 0.0

        r = self.streamer_regime_stats[(streamer, regime)]
        r["n"] += 1.0
        r["w"] += 1.0 if won else 0.0

        reward = 1.0 if won else -1.0
        self.regime_edge[regime] = 0.9 * self.regime_edge[regime] + 0.1 * reward


class Slrabbit99PropFamilyGuardSizerV2(SmartBetStrategyBase):
    """
    slrabbit99-specific family-aware sizing strategy.
    Runtime state is rebuilt from resolved SQLite history on each decision.
    """

    name = "slrabbit99_prop_family_guard_sizer_v2"

    def __init__(self):
        self._cached_signature = None
        self._cached_state = _ReplayState()
        self.crowd_weight = 0.24
        self.regime_weight = 0.16
        self.base_amount = 50000.0
        self.ev_mult = 40000.0
        self.regime_mult = 10000.0
        self.hr_boost = 1.12
        self.hr_ratio = 0.84
        self.er_3_mult = 0.40
        self.er_5_mult = 0.25
        self.hits_2p5_mult = 0.25
        self.hits_5_mult = 0.10
        self.other_3_mult = 0.60
        self.special_ratio_pen = 0.75

    def _db_path(self, bet_obj):
        try:
            return bet_obj._db_path()
        except Exception:
            return "bet_logs.sqlite"

    @staticmethod
    def _odds_regime(odds: float) -> str:
        if odds <= 1.8:
            return "A_1p8"
        if odds <= 2.2:
            return "B_2p2"
        if odds <= 3.0:
            return "C_3p0"
        if odds <= 5.0:
            return "D_5p0"
        return "E_gt5"

    @staticmethod
    def _family(title: str) -> str:
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

    @staticmethod
    def _crowd_edge(ratio: float) -> float:
        return 1.0 - min(2.0, ratio)

    def _load_state(self, bet_obj):
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        if not streamer:
            return self._cached_state

        try:
            db_path = self._db_path(bet_obj)
            with sqlite3.connect(db_path) as conn:
                count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM bet_events
                    WHERE lower(streamer) = ?
                      AND result_type IN ('WIN', 'LOSE')
                      AND COALESCE(winning_outcome_id, '') <> ''
                      AND COALESCE(selected_outcome_id, '') <> ''
                    """,
                    (streamer,),
                ).fetchone()[0]
                signature = (db_path, streamer, int(count))
                if signature == self._cached_signature:
                    return self._cached_state

                rows = conn.execute(
                    """
                    SELECT
                        e.selected_outcome_id,
                        e.winning_outcome_id,
                        o.outcome_odds,
                        COALESCE(e.resolved_at, e.created_at) AS ts,
                        e.event_id
                    FROM bet_events e
                    JOIN bet_outcomes o
                      ON o.event_id = e.event_id
                     AND o.outcome_id = e.selected_outcome_id
                    WHERE lower(e.streamer) = ?
                      AND e.result_type IN ('WIN', 'LOSE')
                      AND COALESCE(e.winning_outcome_id, '') <> ''
                      AND COALESCE(e.selected_outcome_id, '') <> ''
                    ORDER BY ts, e.event_id
                    """,
                    (streamer,),
                ).fetchall()
        except Exception:
            return self._cached_state

        state = _ReplayState()
        for selected_outcome_id, winning_outcome_id, odds, _ts, _event_id in rows:
            odds = float(odds or 1.01)
            won = str(selected_outcome_id or "") == str(winning_outcome_id or "")
            state.update(streamer, odds, won)

        self._cached_signature = signature
        self._cached_state = state
        return state

    def _score(self, title: str, state: _ReplayState, odds: float, ev: float, ratio: float) -> float:
        crowd_edge = self._crowd_edge(ratio)
        rg = self._odds_regime(odds)
        score = ev + self.crowd_weight * crowd_edge + self.regime_weight * state.regime_edge[rg]

        family = self._family(title)
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
        return score, family, rg

    def choose(self, bet_obj, details):
        state = self._load_state(bet_obj)
        title = getattr(bet_obj, "title", "") or ""

        best_idx = 0
        best_score = -10**9
        for i, d in enumerate(details):
            odds = float(d.get("odds", 1.01) or 1.01)
            ev = float(d.get("ev", d.get("ev_raw", 0.0)) or 0.0)
            ratio = float(d.get("ratio", 1.0) or 1.0)
            score, _, _ = self._score(title, state, odds, ev, ratio)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        state = self._load_state(bet_obj)
        title = getattr(bet_obj, "title", "") or ""
        odds = float(selected_detail.get("odds", 1.01) or 1.01)
        ev = float(selected_detail.get("ev", selected_detail.get("ev_raw", 0.0)) or 0.0)
        ratio = float(selected_detail.get("ratio", 1.0) or 1.0)
        family = self._family(title)
        rg = self._odds_regime(odds)

        amount = self.base_amount + self.ev_mult * ev + self.regime_mult * state.regime_edge[rg]
        if family == "CHALLENGE":
            pass
        elif family == "HR_PROP":
            if 2.0 <= odds <= 4.0:
                amount *= self.hr_boost
            if ratio > 1.0:
                amount *= self.hr_ratio
        elif family == "ER_PROP":
            if odds > 3.0:
                amount *= self.er_3_mult
            if odds > 5.0:
                amount *= self.er_5_mult
        elif family == "HITS_PROP":
            if odds > 2.5:
                amount *= self.hits_2p5_mult
            if odds > 5.0:
                amount *= self.hits_5_mult
        else:
            if odds > 3.0:
                amount *= self.other_3_mult

        if ratio > 1.0 and family in {"ER_PROP", "HITS_PROP"}:
            amount *= self.special_ratio_pen

        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "slrabbit99_prop_family_guard_sizer_v2", "tier_slrabbit99_family_guard"
