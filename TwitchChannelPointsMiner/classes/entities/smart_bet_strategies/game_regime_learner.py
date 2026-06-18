import sqlite3
from collections import defaultdict

from .base import SmartBetStrategyBase


def _cluster_title(title: str) -> str:
    title_norm = (title or "").lower()

    if any(k in title_norm for k in ["lol", "英雄聯盟", "勝敗", "輸贏", "潘森", "這把"]):
        return "REGIME_LOL"
    if any(k in title_norm for k in ["tft", "戰棋", "棋", "第幾"]):
        return "REGIME_TFT"
    if any(k in title_norm for k in ["大富翁", "彈珠", "賭", "過關", "幾場"]):
        return "REGIME_LUCK"
    if any(k in title_norm for k in ["聊", "唱歌", "日常"]):
        return "REGIME_CHAT"
    return "REGIME_VARIETY"


class _RegimeReplay:
    def __init__(self):
        self.stats = defaultdict(lambda: {
            "r1_n": 0.0,
            "r1_w": 0.0,
            "r2_n": 0.0,
            "r2_w": 0.0,
        })


class GameRegimeLearner(SmartBetStrategyBase):
    """
    Runtime counterpart to the backtest game_regime_learner strategy.
    Rebuilds streamer x regime stats from resolved SQLite history on each decision.
    """

    name = "game_regime_learner"

    def __init__(self):
        self._cached_signature = None
        self._cached_state = _RegimeReplay()

    def _db_path(self, bet_obj):
        try:
            return bet_obj._db_path()
        except Exception:
            return "bet_logs.sqlite"

    @staticmethod
    def _get_rank_edge(stats, rank, odds):
        if rank == 1:
            wr = (stats["r1_w"] + 3.0 * 0.6) / (stats["r1_n"] + 3.0)
        else:
            wr = (stats["r2_w"] + 3.0 * 0.35) / (stats["r2_n"] + 3.0)
        return wr * odds - 1.0

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
                        e.event_id,
                        e.winning_outcome_id,
                        e.main_title,
                        o.outcome_id,
                        o.outcome_odds,
                        o.outcome_idx,
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
            return self._cached_state

        state = _RegimeReplay()
        current_event_id = None
        event_rows = []

        def flush_event(rows):
            if not rows:
                return
            regime = _cluster_title(rows[0][2])
            key = (streamer, regime)
            ordered = sorted(rows, key=lambda r: float(r[4] or 0.0))
            winner_id = str(rows[0][1] or "")
            stats = state.stats[key]
            stats["r1_n"] += 1.0
            stats["r2_n"] += 1.0
            if ordered and str(ordered[0][3] or "") == winner_id:
                stats["r1_w"] += 1.0
            elif len(ordered) > 1 and str(ordered[1][3] or "") == winner_id:
                stats["r2_w"] += 1.0

        for row in rows:
            event_id = row[0]
            if current_event_id is None:
                current_event_id = event_id
            if event_id != current_event_id:
                flush_event(event_rows)
                event_rows = []
                current_event_id = event_id
            event_rows.append(row)
        flush_event(event_rows)

        self._cached_signature = signature
        self._cached_state = state
        return state

    def choose(self, bet_obj, details):
        state = self._load_state(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        regime = _cluster_title(getattr(bet_obj, "title", "") or "")
        key = (streamer, regime)
        stats = state.stats[key]

        ordered = sorted(list(enumerate(details)), key=lambda pair: float(pair[1].get("odds", 1.01) or 1.01))
        best_idx = ordered[0][0]
        best_edge = -10**9
        for rank_pos, (idx, d) in enumerate(ordered[:2], start=1):
            odds = float(d.get("odds", 1.01) or 1.01)
            edge = self._get_rank_edge(stats, rank_pos, odds)
            if edge > best_edge:
                best_edge = edge
                best_idx = idx
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        state = self._load_state(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        regime = _cluster_title(getattr(bet_obj, "title", "") or "")
        key = (streamer, regime)
        stats = state.stats[key]

        details = getattr(bet_obj, "outcomes", []) or []
        ordered = sorted(details, key=lambda o: float(o.get("odds", 1.01) or 1.01))
        selected_id = str(selected_detail.get("id", selected_detail.get("outcome_id", "")))
        rank = 1
        if len(ordered) > 1 and str(ordered[1].get("id", ordered[1].get("outcome_id", ""))) == selected_id:
            rank = 2

        odds = float(selected_detail.get("odds", 1.01) or 1.01)
        edge = self._get_rank_edge(stats, rank, odds)
        amount = 40000 + 100000 * max(0.0, edge)

        n_total = stats["r1_n"] + stats["r2_n"]
        if n_total < 3:
            amount *= 0.6

        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "game_regime_learner", "tier_game_regime"
