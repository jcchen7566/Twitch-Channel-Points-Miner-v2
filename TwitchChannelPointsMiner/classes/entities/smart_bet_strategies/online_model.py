import math
import sqlite3
from collections import defaultdict

from .base import SmartBetStrategyBase


class _ReplayStats:
    def __init__(self):
        self.global_stats = {"n": 0.0, "w": 0.0}
        self.streamer_stats = defaultdict(lambda: {"n": 0.0, "w": 0.0})
        self.streamer_regime_stats = defaultdict(lambda: {"n": 0.0, "w": 0.0})

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

    def prior_win_rate(self, streamer: str, odds: float) -> float:
        regime = self._odds_regime(odds)
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
        regime = self._odds_regime(odds)
        self.global_stats["n"] += 1.0
        self.global_stats["w"] += 1.0 if won else 0.0

        s = self.streamer_stats[streamer]
        s["n"] += 1.0
        s["w"] += 1.0 if won else 0.0

        r = self.streamer_regime_stats[(streamer, regime)]
        r["n"] += 1.0
        r["w"] += 1.0 if won else 0.0


class OnlineModel(SmartBetStrategyBase):
    """
    Runtime counterpart to the online_model backtest strategy.
    Rebuilds its weights from resolved rows on each decision for restart safety.
    """

    name = "online_model"

    def __init__(self):
        self._cached_signature = None
        self._cached_weights = [0.0] * 6
        self._cached_stats = _ReplayStats()
        self.lr = 0.03
        self.reg = 1e-4
        self.low_p = 0.45
        self.base_amount = 80000
        self.ev_mult = 140000
        self.high_odds_pen = 0.7

    @staticmethod
    def _confidence(sample_size: float, odds: float) -> float:
        if sample_size <= 0:
            return 0.5
        size_factor = min(1.0, math.log1p(sample_size) / 3.5)
        odds_factor = 1.0 if odds <= 3.0 else max(0.70, 1.0 - ((odds - 3.0) * 0.04))
        return max(0.35, min(1.0, size_factor * odds_factor))

    def _db_path(self, bet_obj):
        try:
            return bet_obj._db_path()
        except Exception:
            return "bet_logs.sqlite"

    def _feature_vector(self, prior: float, odds: float, ratio: float, confidence: float, ev: float):
        return [1.0, prior, math.log(max(1.01, odds)), ratio, confidence, ev]

    def _predict(self, weights, feat):
        z = sum(a * b for a, b in zip(weights, feat))
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _load_model(self, bet_obj):
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
                      AND COALESCE(selected_outcome_id, '') <> ''
                    """,
                    (streamer,),
                ).fetchone()[0]
                signature = (db_path, streamer, int(count))
                if signature == self._cached_signature:
                    return self._cached_weights, self._cached_stats

                rows = conn.execute(
                    """
                    SELECT
                        e.selected_outcome_id,
                        e.winning_outcome_id,
                        o.outcome_odds,
                        o.outcome_users_percent,
                        o.total_users,
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
            return [0.0] * 6, _ReplayStats()

        weights = [0.0] * 6
        stats = _ReplayStats()
        for selected_outcome_id, winning_outcome_id, odds, users_pct, total_users, _ts, _event_id in rows:
            odds = float(odds or 1.01)
            users_pct = float(users_pct or 0.0)
            total_users = float(total_users or 0.0)
            ratio = (users_pct / (100.0 / odds)) if odds > 0 else 1.0
            prior = stats.prior_win_rate(streamer, odds)
            confidence = self._confidence(total_users, odds)
            ev_proxy = prior * odds - 1.0
            feat = self._feature_vector(prior, odds, ratio, confidence, ev_proxy)
            p = max(0.02, min(0.98, self._predict(weights, feat)))
            won = str(selected_outcome_id or "") == str(winning_outcome_id or "")
            y = 1.0 if won else 0.0
            g = p - y
            for i in range(len(weights)):
                weights[i] -= self.lr * (g * feat[i] + self.reg * weights[i])
            stats.update(streamer, odds, won)

        self._cached_signature = signature
        self._cached_weights = weights
        self._cached_stats = stats
        return weights, stats

    def choose(self, bet_obj, details):
        weights, stats = self._load_model(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()

        best_idx = 0
        best_ev = -10**9
        for i, d in enumerate(details):
            odds = float(d.get("odds", 1.01))
            prior = stats.prior_win_rate(streamer, odds)
            ratio = float(d.get("ratio", 1.0))
            confidence = float(d.get("confidence", 0.5))
            ev_proxy = prior * odds - 1.0
            feat = self._feature_vector(prior, odds, ratio, confidence, ev_proxy)
            p = max(0.02, min(0.98, self._predict(weights, feat)))
            ev = p * odds - 1.0
            if ev > best_ev:
                best_ev = ev
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        weights, stats = self._load_model(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()

        odds = float(selected_detail.get("odds", 1.01))
        prior = stats.prior_win_rate(streamer, odds)
        ratio = float(selected_detail.get("ratio", 1.0))
        confidence = float(selected_detail.get("confidence", 0.5))
        ev_proxy = prior * odds - 1.0
        feat = self._feature_vector(prior, odds, ratio, confidence, ev_proxy)
        p = max(0.02, min(0.98, self._predict(weights, feat)))
        ev = p * odds - 1.0

        amount = self.base_amount + self.ev_mult * ev
        if p < self.low_p:
            amount *= 0.8
        if odds > 5.0:
            amount *= self.high_odds_pen

        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "online_model", "tier_online_model"
