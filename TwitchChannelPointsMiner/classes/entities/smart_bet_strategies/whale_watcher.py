import sqlite3
from collections import defaultdict

from .base import SmartBetStrategyBase


class _WhaleReplay:
    def __init__(self):
        self.bias = defaultdict(float)
        self.counts = defaultdict(int)


class WhaleWatcher(SmartBetStrategyBase):
    """
    Runtime counterpart to the backtest whale_watcher strategy.
    Rebuilds streamer bias from resolved SQLite history on each decision.
    """

    name = "whale_watcher"

    def __init__(self):
        self._cached_signature = None
        self._cached_state = _WhaleReplay()
        self.lr = 0.15
        self.base_amount = 50000.0
        self.ev_mult = 80000.0

    def _db_path(self, bet_obj):
        try:
            return bet_obj._db_path()
        except Exception:
            return "bet_logs.sqlite"

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
                    """,
                    (streamer,),
                ).fetchone()[0]
                signature = (db_path, streamer, int(count))
                if signature == self._cached_signature:
                    return self._cached_state

                rows = conn.execute(
                    """
                    SELECT
                        e.winning_outcome_id,
                        o.outcome_id,
                        o.outcome_users_percent,
                        o.outcome_points_percent,
                        COALESCE(e.resolved_at, e.created_at) AS ts,
                        e.event_id
                    FROM bet_events e
                    JOIN bet_outcomes o
                      ON o.event_id = e.event_id
                     AND o.outcome_id = e.winning_outcome_id
                    WHERE lower(e.streamer) = ?
                      AND e.result_type IN ('WIN', 'LOSE')
                      AND COALESCE(e.winning_outcome_id, '') <> ''
                    ORDER BY ts, e.event_id
                    """,
                    (streamer,),
                ).fetchall()
        except Exception:
            return self._cached_state

        state = _WhaleReplay()
        for _winner_id, _outcome_id, users_pct, points_pct, _ts, _event_id in rows:
            div = float(users_pct or 0.0) - float(points_pct or 0.0)
            state.bias[streamer] += self.lr * (div - state.bias[streamer])
            state.counts[streamer] += 1

        self._cached_signature = signature
        self._cached_state = state
        return state

    @staticmethod
    def _outcome_map(bet_obj):
        out = {}
        for o in getattr(bet_obj, "outcomes", []) or []:
            oid = str(o.get("id", o.get("outcome_id", "")))
            out[oid] = o
        return out

    def _divergence(self, raw_outcome):
        users_pct = float(raw_outcome.get("percentage_users", raw_outcome.get("users_pct", 0.0)) or 0.0)
        points_pct = float(raw_outcome.get("odds_percentage", raw_outcome.get("points_pct", 0.0)) or 0.0)
        return users_pct - points_pct

    def choose(self, bet_obj, details):
        state = self._load_state(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        bias = state.bias[streamer]
        raw_map = self._outcome_map(bet_obj)

        best_idx = 0
        best_score = -10**9
        for i, d in enumerate(details):
            oid = str(d.get("id", d.get("outcome_id", "")))
            raw = raw_map.get(oid, {})
            ev = float(d.get("ev", d.get("ev_raw", 0.0)) or 0.0)
            divergence = self._divergence(raw)
            trust_score = -bias * divergence
            score = ev + 0.05 * trust_score
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        state = self._load_state(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        bias = state.bias[streamer]
        raw_map = self._outcome_map(bet_obj)

        oid = str(selected_detail.get("id", selected_detail.get("outcome_id", "")))
        raw = raw_map.get(oid, {})
        ev = float(selected_detail.get("ev", selected_detail.get("ev_raw", 0.0)) or 0.0)
        divergence = self._divergence(raw)

        amount = self.base_amount + self.ev_mult * ev
        if (bias < 0 and divergence < -2.0) or (bias > 0 and divergence > 2.0):
            amount += 25000

        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "whale_watcher", "tier_whale_watcher"
