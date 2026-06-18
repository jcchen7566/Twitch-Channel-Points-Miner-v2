import math
import sqlite3

from .base import SmartBetStrategyBase


class Slrabbit99CrowdFadeMomentumV2(SmartBetStrategyBase):
    """
    slrabbit99-specific crowd-fade strategy tuned from streamer-only backtests.
    Runtime state is rebuilt from resolved log history each decision.
    """

    name = "slrabbit99_crowd_fade_momentum_v2"

    def __init__(self):
        self._cached_regime_edge = None

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

    def _empty_edge(self):
        return {k: 0.0 for k in ("A_1p8", "B_2p2", "C_3p0", "D_5p0", "E_gt5")}

    def _load_regime_edge(self, bet_obj):
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        if streamer != "slrabbit99":
            return self._empty_edge()

        try:
            db_path = bet_obj._db_path()
        except Exception:
            return self._empty_edge()

        sql = """
        SELECT
          o.outcome_odds,
          CASE
            WHEN e.selected_outcome_id = e.winning_outcome_id THEN 1.0
            ELSE -1.0
          END AS reward
        FROM bet_events e
        JOIN bet_outcomes o
          ON o.event_id = e.event_id
         AND o.outcome_id = e.selected_outcome_id
        WHERE lower(e.streamer) = ?
          AND e.result_type IN ('WIN', 'LOSE')
          AND COALESCE(e.winning_outcome_id, '') <> ''
        ORDER BY COALESCE(e.resolved_at, e.created_at), e.event_id
        """

        edge = self._empty_edge()
        try:
            with sqlite3.connect(db_path) as conn:
                for odds, reward in conn.execute(sql, (streamer,)).fetchall():
                    rg = self._odds_regime(float(odds or 1.01))
                    edge[rg] = 0.9 * edge[rg] + 0.1 * float(reward or -1.0)
        except Exception:
            return self._empty_edge()
        return edge

    def choose(self, bet_obj, details):
        edge = self._load_regime_edge(bet_obj)
        self._cached_regime_edge = edge

        best_index = 0
        best_score = -10**9
        for i, d in enumerate(details):
            ratio = float(d.get("ratio", 1.0))
            ev = float(d.get("ev", 0.0))
            odds = float(d.get("odds", 1.01))
            rg = self._odds_regime(odds)
            crowd_edge = 1.0 - min(2.0, ratio)
            score = ev + 0.16 * crowd_edge + 0.14 * edge[rg]
            if score > best_score:
                best_score = score
                best_index = i
        return best_index

    def size(self, bet_obj, balance, selected_detail):
        edge = self._cached_regime_edge or self._load_regime_edge(bet_obj)
        odds = float(selected_detail.get("odds", 1.01))
        ev = float(selected_detail.get("ev", 0.0))
        ratio = float(selected_detail.get("ratio", 1.0))
        rg = self._odds_regime(odds)

        amount = 80000 + 50000 * ev + 40000 * edge[rg]
        if ratio > 1.0:
            amount *= 0.7
        if odds > 3.0:
            amount *= 0.6

        amount = int(amount)
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "slrabbit99_crowd_fade_momentum_v2", "tier_slrabbit99"
