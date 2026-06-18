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


class Slrabbit99MarginSoftmaxV1(SmartBetStrategyBase):
    """
    Runtime counterpart to the backtest slrabbit99_margin_softmax_v1 strategy.
    Replays resolved rows from SQLite on each decision, then uses the learned
    softmax weights to score the current event outcomes.
    """

    name = "slrabbit99_margin_softmax_v1"

    def __init__(self):
        self.lr = 0.035
        self.reg = 1e-4
        self.base_bet = 20000.0
        self.edge_mult = 100000.0
        self._cached_signature = None
        self._cached_weights = None
        self._cached_stats = None
        self._initial_weights = [
            -0.10,
            0.10,
            0.00,
            0.00,
            -0.04,
            -0.10,
            0.00,
            -0.06,
            0.00,
            0.00,
            0.08,
            0.06,
            0.00,
            -0.04,
            -0.02,
            0.02,
            0.00,
        ]

    @staticmethod
    def _db_path(bet_obj):
        try:
            return bet_obj._db_path()
        except Exception:
            return "bet_logs.sqlite"

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x < -35.0:
            return 0.0
        if x > 35.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _softmax(values):
        if not values:
            return []
        peak = max(values)
        exps = [math.exp(max(-20.0, min(20.0, v - peak))) for v in values]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

    @staticmethod
    def _feature_vector(outcome, prior: float):
        odds = float(outcome.get("odds", 1.01) or 1.01)
        users_pct = float(outcome.get("percentage_users", outcome.get("outcome_users_percent", 0.0)) or 0.0)
        points_pct = float(outcome.get("odds_percentage", outcome.get("outcome_points_percent", 0.0)) or 0.0)
        total_users = float(outcome.get("total_users", 0.0) or 0.0)
        total_points = float(outcome.get("total_points", 0.0) or 0.0)
        ratio = (users_pct / points_pct) if points_pct > 0 else 1.0
        log_odds = math.log(max(1.01, odds))
        log_liq = math.log10(max(1.0, total_points))
        crowd = points_pct / 100.0
        implied = 1.0 / odds
        surprise = crowd - implied
        conf = 1.0 / (1.0 + math.exp(-((total_users) - 15.0) / 5.0))
        ev = prior * odds - 1.0
        return {
            "bias": 1.0,
            "log_odds": log_odds,
            "users_pct": users_pct / 100.0,
            "points_pct": points_pct / 100.0,
            "ratio": ratio,
            "rel_ratio": 0.0,
            "rel_odds": 0.0,
            "rel_surprise": 0.0,
            "log_liq": log_liq,
            "p": max(0.01, min(0.99, prior)),
            "ev": ev,
            "conf": conf,
            "momentum": surprise * prior,
            "crowd_liq": ratio * log_liq / 10.0,
            "users_minus_points": (users_pct - points_pct) / 100.0,
            "points_minus_users": (points_pct - users_pct) / 100.0,
            "efficiency": surprise,
        }

    def _score_outcome(self, weights, feat):
        names = [
            "bias",
            "log_odds",
            "users_pct",
            "points_pct",
            "ratio",
            "rel_ratio",
            "rel_odds",
            "rel_surprise",
            "log_liq",
            "p",
            "ev",
            "conf",
            "momentum",
            "crowd_liq",
            "users_minus_points",
            "points_minus_users",
            "efficiency",
        ]
        return sum(w * feat[name] for w, name in zip(weights, names))

    def _load_model(self, bet_obj):
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        if not streamer:
            return list(self._initial_weights), _ReplayStats()

        try:
            db_path = self._db_path(bet_obj)
        except Exception:
            return list(self._initial_weights), _ReplayStats()

        try:
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
                    return self._cached_weights, self._cached_stats

                rows = conn.execute(
                    """
                    SELECT
                        e.event_id,
                        e.selected_outcome_id,
                        e.winning_outcome_id,
                        e.result_type,
                        COALESCE(e.resolved_at, e.created_at) AS ts,
                        o.outcome_idx,
                        o.outcome_id,
                        o.outcome_odds,
                        o.outcome_users_percent,
                        o.outcome_points_percent,
                        o.total_users,
                        o.total_points
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
            return list(self._initial_weights), _ReplayStats()

        weights = list(self._initial_weights)
        stats = _ReplayStats()
        current_event_id = None
        event_rows = []

        def flush_event(rows):
            if not rows:
                return
            local_streamer = streamer
            outcomes = []
            for _, selected_outcome_id, winning_outcome_id, _result_type, _ts, _idx, outcome_id, odds, users_pct, points_pct, total_users, total_points in rows:
                outcomes.append({
                    "outcome_id": str(outcome_id or ""),
                    "odds": float(odds or 1.01),
                    "percentage_users": float(users_pct or 0.0),
                    "odds_percentage": float(points_pct or 0.0),
                    "total_users": float(total_users or 0.0),
                    "total_points": float(total_points or 0.0),
                })
            if not outcomes:
                return
            winner_id = str(rows[0][2] or "")
            features = []
            for out in outcomes:
                prior = stats.prior_win_rate(local_streamer, out["odds"])
                features.append(self._feature_vector(out, prior))
            scores = [self._score_outcome(weights, f) for f in features]
            probs = self._softmax(scores)
            y = [1.0 if out["outcome_id"] == winner_id else 0.0 for out in outcomes]
            grad = [p - yy for p, yy in zip(probs, y)]
            for i, name in enumerate([
                "bias", "log_odds", "users_pct", "points_pct", "ratio", "rel_ratio", "rel_odds",
                "rel_surprise", "log_liq", "p", "ev", "conf", "momentum", "crowd_liq",
                "users_minus_points", "points_minus_users", "efficiency",
            ]):
                g = sum(grad[j] * features[j][name] for j in range(len(outcomes)))
                weights[i] -= self.lr * (g + self.reg * weights[i])
            # Update with the actual winner from the historical event.
            winner_outcome = next((o for o in outcomes if o["outcome_id"] == winner_id), None)
            if winner_outcome is not None:
                stats.update(local_streamer, winner_outcome["odds"], True)

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
        self._cached_weights = weights
        self._cached_stats = stats
        return weights, stats

    def _event_features(self, bet_obj, details, prior_stats):
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        outcomes = getattr(bet_obj, "outcomes", []) or []
        vectors = []
        scores = []
        weights, _ = self._load_model(bet_obj)
        for d in details:
            odds = float(d.get("odds", 1.01) or 1.01)
            prior = prior_stats.prior_win_rate(streamer, odds)
            outcome = None
            outcome_id = d.get("id", d.get("outcome_id", ""))
            for raw in outcomes:
                if str(raw.get("id", raw.get("outcome_id", ""))) == str(outcome_id):
                    outcome = raw
                    break
            if outcome is None:
                outcome = {
                    "odds": odds,
                    "percentage_users": float(d.get("users_pct", d.get("ratio", 1.0)) or 0.0),
                    "odds_percentage": float(d.get("points_pct", 100.0 / odds) or 0.0),
                    "total_users": float(d.get("total_users", 0.0) or 0.0),
                    "total_points": float(d.get("total_points", 0.0) or 0.0),
                }
            feat = self._feature_vector(outcome, prior)
            vectors.append(feat)
            scores.append(self._score_outcome(weights, feat))
        probs = self._softmax(scores)
        return vectors, scores, probs, weights, prior_stats

    def choose(self, bet_obj, details):
        weights, prior_stats = self._load_model(bet_obj)
        vectors, scores, probs, _, _ = self._event_features(bet_obj, details, prior_stats)

        best_idx = 0
        best_edge = -10**9
        for i, d in enumerate(details):
            odds = float(d.get("odds", 1.01) or 1.01)
            edge = probs[i] * odds - 1.0
            if edge > best_edge:
                best_edge = edge
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        weights, prior_stats = self._load_model(bet_obj)
        outcomes = getattr(bet_obj, "outcomes", []) or []
        details = []
        for o in outcomes:
            details.append({
                "id": o.get("id", ""),
                "odds": float(o.get("odds", 1.01) or 1.01),
                "users_pct": float(o.get("percentage_users", 0.0) or 0.0),
                "points_pct": float(o.get("odds_percentage", 100.0 / max(1.01, float(o.get("odds", 1.01) or 1.01))) or 0.0),
                "total_users": float(o.get("total_users", 0.0) or 0.0),
                "total_points": float(o.get("total_points", 0.0) or 0.0),
            })
        vectors, scores, probs, _, _ = self._event_features(bet_obj, details, prior_stats)
        selected_id = selected_detail.get("id", selected_detail.get("outcome_id", ""))
        selected_index = 0
        for i, d in enumerate(details):
            if str(d.get("id", "")) == str(selected_id):
                selected_index = i
                break

        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        edge = probs[selected_index] * float(selected_detail.get("odds", 1.01) or 1.01) - 1.0
        amount = self.base_bet + self.edge_mult * max(0.0, edge) * self._sigmoid(margin)
        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "slrabbit99_margin_softmax_v1", "tier_slrabbit99_margin"
