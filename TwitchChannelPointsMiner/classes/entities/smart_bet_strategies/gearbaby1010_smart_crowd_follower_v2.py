import sqlite3
from collections import defaultdict

from .base import SmartBetStrategyBase


class _CrowdReplay:
    def __init__(self):
        self.correct = 0.0
        self.total = 0.0

    def acc(self) -> float:
        return (self.correct + 1.0) / (self.total + 2.0)


class Gearbaby1010SmartCrowdFollowerV2(SmartBetStrategyBase):
    """
    gearbaby1010-specific crowd follower tuned from streamer-only backtests.
    Runtime state is rebuilt from resolved log history each decision.
    """

    name = "gearbaby1010_smart_crowd_follower_v2"

    def __init__(self):
        self._cached_signature = None
        self._cached_crowd = defaultdict(_CrowdReplay)
        self.crowd_weight = 0.12
        self.surprise_weight = 0.12
        self.base_amount = 70000
        self.ev_mult = 110000
        self.ratio_cut = 0.95
        self.ratio_mult = 0.85
        self.odds_cut = 6.0
        self.odds_mult = 0.7
        self.crowd_bonus = 0.0

    def _db_path(self, bet_obj):
        try:
            return bet_obj._db_path()
        except Exception:
            return "bet_logs.sqlite"

    def _load_crowd(self, bet_obj):
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
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
                    """,
                    (streamer,),
                ).fetchone()[0]
                signature = (db_path, streamer, int(count))
                if signature == self._cached_signature:
                    return self._cached_crowd

                rows = conn.execute(
                    """
                    SELECT
                        e.event_id,
                        e.winning_outcome_id,
                        o.outcome_id,
                        o.outcome_users_percent,
                        COALESCE(e.resolved_at, e.created_at) AS ts
                    FROM bet_events e
                    JOIN bet_outcomes o
                      ON o.event_id = e.event_id
                    WHERE lower(e.streamer) = ?
                      AND e.result_type IN ('WIN', 'LOSE')
                      AND COALESCE(e.winning_outcome_id, '') <> ''
                    ORDER BY ts, e.event_id, o.outcome_idx
                    """,
                    (streamer,),
                ).fetchall()
        except Exception:
            return defaultdict(_CrowdReplay)

        crowd = defaultdict(_CrowdReplay)
        current_event_id = None
        event_rows = []

        def flush_event(rows):
            if not rows:
                return
            winner = str(rows[0][1] or "")
            crowd_favorite = max(rows, key=lambda r: float(r[3] or 0.0))[2]
            replay = crowd[streamer]
            replay.total += 1.0
            if str(crowd_favorite or "") == winner:
                replay.correct += 1.0

        for event_id, winning_outcome_id, outcome_id, outcome_users_percent, _ts in rows:
            if current_event_id is None:
                current_event_id = event_id
            if event_id != current_event_id:
                flush_event(event_rows)
                event_rows = []
                current_event_id = event_id
            event_rows.append((event_id, winning_outcome_id, outcome_id, outcome_users_percent, _ts))
        flush_event(event_rows)

        self._cached_signature = signature
        self._cached_crowd = crowd
        return crowd

    def _acc(self, crowd, streamer: str) -> float:
        return crowd[streamer].acc() if streamer in crowd else 0.5

    def _score(self, ev: float, ratio: float, confidence: float, acc: float) -> float:
        crowd_term = self.crowd_weight * (confidence - 0.5)
        surprise_term = -self.surprise_weight * max(0.0, ratio - self.ratio_cut)
        return (1.0 - acc) * ev + acc * (crowd_term + surprise_term)

    def choose(self, bet_obj, details):
        crowd = self._load_crowd(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        acc = self._acc(crowd, streamer)

        best_idx = 0
        best_score = -10**9
        for i, d in enumerate(details):
            ev = float(d.get("ev", 0.0))
            ratio = float(d.get("ratio", 1.0))
            confidence = float(d.get("confidence", 0.5))
            score = self._score(ev, ratio, confidence, acc)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        crowd = self._load_crowd(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        _ = self._acc(crowd, streamer)

        ev = float(selected_detail.get("ev", 0.0))
        ratio = float(selected_detail.get("ratio", 1.0))
        odds = float(selected_detail.get("odds", 1.01))

        amount = self.base_amount + self.ev_mult * ev
        if ratio > self.ratio_cut:
            amount *= self.ratio_mult
        if odds > self.odds_cut:
            amount *= self.odds_mult
        if self.crowd_bonus and ratio <= self.ratio_cut:
            amount += self.crowd_bonus

        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "gearbaby1010_smart_crowd_follower_v2", "tier_gearbaby1010_crowd"
