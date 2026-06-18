import math
import sqlite3
from .base import SmartBetStrategyBase


class SoftmaxBanditV1(SmartBetStrategyBase):
    """
    Runtime softmax aligned with backtest baseline parameters,
    using a single data path: rebuild arms from SQLite each decision.
    """

    name = "softmax_bandit_v1"

    def __init__(self):
        self.tau = 0.15

    def _get_arm_key(self, odds):
        if odds <= 1.8:
            return "A"
        if odds <= 2.2:
            return "B"
        if odds <= 3.0:
            return "C"
        if odds <= 5.0:
            return "D"
        return "E"

    @staticmethod
    def _empty_arms():
        return {k: {"n": 0, "avg": 0.0} for k in ("A", "B", "C", "D", "E")}

    def _load_streamer_arms(self, bet_obj):
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        if not streamer:
            return self._empty_arms()

        try:
            db_path = bet_obj._db_path()
        except Exception:
            return self._empty_arms()

        arms = self._empty_arms()
        sql = """
        SELECT
          CASE
            WHEN o.outcome_odds <= 1.8 THEN 'A'
            WHEN o.outcome_odds <= 2.2 THEN 'B'
            WHEN o.outcome_odds <= 3.0 THEN 'C'
            WHEN o.outcome_odds <= 5.0 THEN 'D'
            ELSE 'E'
          END AS arm,
          COUNT(*) AS n,
          AVG(
            CASE
              WHEN e.selected_outcome_id = e.winning_outcome_id THEN (o.outcome_odds - 1.0)
              ELSE -1.0
            END
          ) AS avg_pnl
        FROM bet_events e
        JOIN bet_outcomes o
          ON o.event_id = e.event_id
         AND o.outcome_id = e.selected_outcome_id
        WHERE lower(e.streamer) = ?
          AND e.result_type IN ('WIN', 'LOSE')
          AND COALESCE(e.winning_outcome_id, '') <> ''
        GROUP BY arm
        """

        try:
            with sqlite3.connect(db_path) as conn:
                for arm, n, avg_pnl in conn.execute(sql, (streamer,)).fetchall():
                    if arm in arms:
                        arms[arm]["n"] = int(n or 0)
                        arms[arm]["avg"] = float(avg_pnl or 0.0)
        except Exception:
            return self._empty_arms()

        return arms

    def choose(self, bet_obj, details):
        best_index = 0
        best_ev = -10**9
        for i, d in enumerate(details):
            if d["ev"] > best_ev:
                best_ev = d["ev"]
                best_index = i
        return best_index

    def size(self, bet_obj, balance, selected_detail):
        odds = selected_detail["odds"]
        ev = selected_detail["ev"]
        arm = self._get_arm_key(odds)

        arms = self._load_streamer_arms(bet_obj)
        vals = [arms[k]["avg"] for k in ("A", "B", "C", "D", "E")]
        mx = max(vals)
        exps = [math.exp(max(-10, min(10, (v - mx) / self.tau))) for v in vals]
        z = sum(exps)
        probs = {k: exps[i] / z for i, k in enumerate(("A", "B", "C", "D", "E"))}
        weight = probs[arm]

        amount = 20000 + 100000 * weight + 50000 * max(-0.2, min(0.4, ev))
        amount = int(amount)
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), f"softmax_{arm}", "tier_softmax"
