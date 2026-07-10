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
    def _feature_vector(outcome, prior: float,
                        hw: float = None, ht_total: float = None,
                        rw: float = None, rt: float = None,
                        all_outcomes: list = None):
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

        # p: exact backtest formula 0.50*hw/ht + 0.20*rw/rt + 0.30*implied
        p_hist_w = (hw / ht_total) if (hw is not None and ht_total is not None and ht_total > 0) else implied
        p_hist_r = (rw / rt) if (rw is not None and rt is not None and rt > 0) else implied
        p = max(0.01, min(0.99, 0.50 * p_hist_w + 0.20 * p_hist_r + 0.30 * implied))
        ev = p * odds - 1.0

        # conf: use ht when available (backtest), else total_users (live fallback)
        conf_denom = ht_total if (ht_total is not None and ht_total > 0) else total_users
        conf = 1.0 / (1.0 + math.exp(-((conf_denom) - 15.0) / 5.0))

        # momentum: use surprise * hw/ht when available, else surprise * prior
        momentum_src = p_hist_w if (hw is not None and ht_total is not None and ht_total > 0) else prior
        momentum = surprise * momentum_src

        # rel_* Z-scores across all outcomes (backtest: event_outcomes=event.outcomes)
        rel_ratio = 0.0
        rel_odds = 0.0
        rel_surprise = 0.0
        if all_outcomes and len(all_outcomes) > 1:
            all_ratios = []
            all_log_odds = []
            all_surprises = []
            for ao in all_outcomes:
                ao_users_pct = float(ao.get("percentage_users", ao.get("outcome_users_percent", ao.get("users_pct", 0.0))) or 0.0)
                ao_points_pct = float(ao.get("odds_percentage", ao.get("outcome_points_percent", ao.get("points_pct", 0.0))) or 0.0)
                ao_odds = float(ao.get("odds", 1.01) or 1.01)
                ao_ratio = (ao_users_pct / ao_points_pct) if ao_points_pct > 0 else 1.0
                all_ratios.append(ao_ratio)
                all_log_odds.append(math.log(max(1.01, ao_odds)))
                ao_implied = 1.0 / ao_odds
                all_surprises.append(ao_points_pct / 100.0 - ao_implied)

            mu_r = sum(all_ratios) / len(all_ratios)
            std_r = math.sqrt(sum((x - mu_r)**2 for x in all_ratios) / len(all_ratios)) + 1e-6
            rel_ratio = (ratio - mu_r) / std_r

            mu_o = sum(all_log_odds) / len(all_log_odds)
            std_o = math.sqrt(sum((x - mu_o)**2 for x in all_log_odds) / len(all_log_odds)) + 1e-6
            rel_odds = (log_odds - mu_o) / std_o

            mu_s = sum(all_surprises) / len(all_surprises)
            std_s = math.sqrt(sum((x - mu_s)**2 for x in all_surprises) / len(all_surprises)) + 1e-6
            rel_surprise = (surprise - mu_s) / std_s

        return {
            "bias": 1.0,
            "log_odds": log_odds,
            "users_pct": users_pct / 100.0,
            "points_pct": points_pct / 100.0,
            "ratio": ratio,
            "rel_ratio": rel_ratio,
            "rel_odds": rel_odds,
            "rel_surprise": rel_surprise,
            "log_liq": log_liq,
            "p": p,
            "ev": ev,
            "conf": conf,
            "momentum": momentum,
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
                        o.total_points,
                        o.history_wins_weighted,
                        o.history_total_weighted,
                        o.history_wins_raw,
                        o.history_total_raw
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
            for _, selected_outcome_id, winning_outcome_id, _result_type, _ts, _idx, outcome_id, odds, users_pct, points_pct, total_users, total_points, hw, ht, rw, rt in rows:
                outcomes.append({
                    "outcome_id": str(outcome_id or ""),
                    "odds": float(odds or 1.01),
                    "percentage_users": float(users_pct or 0.0),
                    "odds_percentage": float(points_pct or 0.0),
                    "total_users": float(total_users or 0.0),
                    "total_points": float(total_points or 0.0),
                    "hw": float(hw or 0.0),
                    "ht_total": float(ht or 0.0),
                    "rw": float(rw or 0.0),
                    "rt": float(rt or 0.0),
                })
            if not outcomes:
                return
            winner_id = str(rows[0][2] or "")
            features = []
            for out in outcomes:
                prior = stats.prior_win_rate(local_streamer, out["odds"])
                hw_val = out["hw"] if out["ht_total"] > 0 else None
                ht_val = out["ht_total"] if out["ht_total"] > 0 else None
                rw_val = out["rw"] if out["rt"] > 0 else None
                rt_val = out["rt"] if out["rt"] > 0 else None
                feat = self._feature_vector(out, prior,
                                            hw=hw_val, ht_total=ht_val,
                                            rw=rw_val, rt=rt_val,
                                            all_outcomes=outcomes)
                features.append(feat)
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
            # Match backtest engine: stats update based on the selected outcome's actual result.
            # Simulate the same selection logic as backtest pick(): outcome with max prob*odds-1.
            best_idx = 0
            best_edge = -10**9
            for i, out in enumerate(outcomes):
                edge = probs[i] * out["odds"] - 1.0
                if edge > best_edge:
                    best_edge = edge
                    best_idx = i
            selected_outcome = outcomes[best_idx]
            won = selected_outcome["outcome_id"] == winner_id
            stats.update(local_streamer, selected_outcome["odds"], won)

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
            feat = self._feature_vector(outcome, prior, all_outcomes=details)
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
