import math
import sqlite3
from collections import defaultdict

from .base import SmartBetStrategyBase


def _cluster_idx(title: str) -> int:
    t = (title or "").strip().lower()
    if any(k in t for k in ["誰會贏", "哪邊會贏", " vs ", "誰贏"]):
        return 0
    if any(k in t for k in ["失分", "勝投", "全壘打", "安打", "觸身", "er ", "er)"]):
        return 1
    if any(k in t for k in ["命一關不刷", "一關不刷"]):
        return 2
    if any(k in t for k in ["第幾名", "開到幾星", "抽"]):
        return 3
    if any(k in t for k in ["會不會過關", "能不能過關", "一把決勝負"]):
        return 4
    if "最穩的那台" in t:
        return 5
    if "丟骨頭" in t:
        return 6
    return 7


CLUSTER_NAMES = [
    "WHO_WIN", "BASEBALL", "LIFE_GAMES", "NUMBER_GUESS",
    "CHALLENGE", "LADDER", "BONE", "CATCH_ALL",
]
N_CLUSTERS = len(CLUSTER_NAMES)

FEATURE_NAMES = [
    "bias", "log_odds", "users_pct", "points_pct", "ratio",
    "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
    "p", "ev", "conf", "momentum", "crowd_liq",
    "users_minus_points", "points_minus_users", "efficiency",
]

_INITIAL_WEIGHTS = [
    -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
    0.00, 0.00, 0.08, 0.06, 0.00, -0.04, -0.02, 0.02, 0.00,
]


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


class Slrabbit99ClusterEnsembleV1(SmartBetStrategyBase):
    name = "slrabbit99_cluster_ensemble_v1"

    def __init__(self):
        self.lr = 0.05
        self.base_bet = 20000.0
        self.edge_mult = 100000.0
        self._cached_signature = None
        self._cached_cluster_weights = None
        self._cached_stats = None

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

        p_hist_w = (hw / ht_total) if (hw is not None and ht_total is not None and ht_total > 0) else implied
        p_hist_r = (rw / rt) if (rw is not None and rt is not None and rt > 0) else implied
        p = max(0.01, min(0.99, 0.50 * p_hist_w + 0.20 * p_hist_r + 0.30 * implied))
        ev = p * odds - 1.0

        conf_denom = ht_total if (ht_total is not None and ht_total > 0) else total_users
        conf = 1.0 / (1.0 + math.exp(-((conf_denom) - 15.0) / 5.0))

        momentum_src = p_hist_w if (hw is not None and ht_total is not None and ht_total > 0) else prior
        momentum = surprise * momentum_src

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
        return sum(w * feat[name] for w, name in zip(weights, FEATURE_NAMES))

    def _ensemble_probs(self, features, cidx):
        scores = [0.0] * len(features)
        for c in range(N_CLUSTERS):
            w = self._cluster_weights[c]
            sim = 0.6 if c == cidx else 0.4 / (N_CLUSTERS - 1 + 1e-10)
            for j in range(len(features)):
                scores[j] += sim * self._score_outcome(w, features[j])
        return self._softmax(scores)

    def _load_model(self, bet_obj):
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        if not streamer:
            self._cluster_weights = [list(_INITIAL_WEIGHTS) for _ in range(N_CLUSTERS)]
            return self._cluster_weights, _ReplayStats()

        try:
            db_path = self._db_path(bet_obj)
        except Exception:
            self._cluster_weights = [list(_INITIAL_WEIGHTS) for _ in range(N_CLUSTERS)]
            return self._cluster_weights, _ReplayStats()

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
                    return self._cached_cluster_weights, self._cached_stats

                rows = conn.execute(
                    """
                    SELECT
                        e.event_id,
                        e.main_title,
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
            self._cluster_weights = [list(_INITIAL_WEIGHTS) for _ in range(N_CLUSTERS)]
            return self._cluster_weights, _ReplayStats()

        cluster_weights = [list(_INITIAL_WEIGHTS) for _ in range(N_CLUSTERS)]
        stats = _ReplayStats()
        current_event_id = None
        current_title = ""
        event_rows = []

        def flush_event(rows_local):
            if not rows_local:
                return
            local_title = rows_local[0][1] or ""
            cidx = _cluster_idx(local_title)
            outcomes = []
            for r in rows_local:
                outcomes.append({
                    "outcome_id": str(r[7] or ""),
                    "odds": float(r[8] or 1.01),
                    "percentage_users": float(r[9] or 0.0),
                    "odds_percentage": float(r[10] or 0.0),
                    "total_users": float(r[11] or 0.0),
                    "total_points": float(r[12] or 0.0),
                    "hw": float(r[13] or 0.0),
                    "ht_total": float(r[14] or 0.0),
                    "rw": float(r[15] or 0.0),
                    "rt": float(r[16] or 0.0),
                })
            if not outcomes:
                return
            winner_id = str(rows_local[0][3] or "")
            features = []
            for out in outcomes:
                prior = stats.prior_win_rate(streamer, out["odds"])
                hw_val = out["hw"] if out["ht_total"] > 0 else None
                ht_val = out["ht_total"] if out["ht_total"] > 0 else None
                rw_val = out["rw"] if out["rt"] > 0 else None
                rt_val = out["rt"] if out["rt"] > 0 else None
                feat = self._feature_vector(out, prior,
                                            hw=hw_val, ht_total=ht_val,
                                            rw=rw_val, rt=rt_val,
                                            all_outcomes=outcomes)
                features.append(feat)
            probs = self._ensemble_probs(features, cidx)
            y = [1.0 if out["outcome_id"] == winner_id else 0.0 for out in outcomes]
            grad = [p - yy for p, yy in zip(probs, y)]
            w = cluster_weights[cidx]
            for i, name in enumerate(FEATURE_NAMES):
                g = sum(grad[j] * features[j][name] for j in range(len(outcomes)))
                w[i] -= self.lr * g

            best_idx = 0
            best_edge = -10**9
            for i, out in enumerate(outcomes):
                edge = probs[i] * out["odds"] - 1.0
                if edge > best_edge:
                    best_edge = edge
                    best_idx = i
            selected_outcome = outcomes[best_idx]
            won = selected_outcome["outcome_id"] == winner_id
            stats.update(streamer, selected_outcome["odds"], won)

        for row in rows:
            event_id = row[0]
            title = row[1] or ""
            if current_event_id is None:
                current_event_id = event_id
                current_title = title
            if event_id != current_event_id:
                flush_event(event_rows)
                event_rows = []
                current_event_id = event_id
                current_title = title
            event_rows.append(row)
        flush_event(event_rows)

        self._cached_signature = signature
        self._cached_cluster_weights = cluster_weights
        self._cached_stats = stats
        self._cluster_weights = cluster_weights
        return cluster_weights, stats

    def _event_features(self, bet_obj, details, prior_stats):
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        cluster_weights, _ = self._load_model(bet_obj)
        cidx = _cluster_idx(getattr(bet_obj, "main_title", ""))
        vectors = []
        for d in details:
            odds = float(d.get("odds", 1.01) or 1.01)
            prior = prior_stats.prior_win_rate(streamer, odds)
            outcome = {
                "odds": odds,
                "percentage_users": float(d.get("users_pct", d.get("ratio", 1.0)) or 0.0),
                "odds_percentage": float(d.get("points_pct", 100.0 / odds) or 0.0),
                "total_users": float(d.get("total_users", 0.0) or 0.0),
                "total_points": float(d.get("total_points", 0.0) or 0.0),
            }
            # Try to match with bet_obj.outcomes for richer features
            bet_outcomes = getattr(bet_obj, "outcomes", []) or []
            for bo in bet_outcomes:
                if str(bo.get("id", bo.get("outcome_id", ""))) == str(d.get("id", d.get("outcome_id", ""))):
                    outcome = {
                        "odds": float(bo.get("odds", odds) or 1.01),
                        "percentage_users": float(bo.get("percentage_users", bo.get("outcome_users_percent", d.get("users_pct", 0.0))) or 0.0),
                        "odds_percentage": float(bo.get("odds_percentage", bo.get("outcome_points_percent", 100.0 / max(1.01, odds))) or 0.0),
                        "total_users": float(bo.get("total_users", 0.0) or 0.0),
                        "total_points": float(bo.get("total_points", 0.0) or 0.0),
                        "hw": float(bo.get("history_wins_weighted", 0.0) or 0.0),
                        "ht_total": float(bo.get("history_total_weighted", 0.0) or 0.0),
                        "rw": float(bo.get("history_wins_raw", 0.0) or 0.0),
                        "rt": float(bo.get("history_total_raw", 0.0) or 0.0),
                    }
                    break
            feat = self._feature_vector(outcome, prior, all_outcomes=details)
            vectors.append(feat)

        scores = [0.0] * len(vectors)
        for c in range(N_CLUSTERS):
            w = cluster_weights[c]
            sim = 0.6 if c == cidx else 0.4 / (N_CLUSTERS - 1 + 1e-10)
            for j in range(len(vectors)):
                scores[j] += sim * self._score_outcome(w, vectors[j])
        probs = self._softmax(scores)
        return vectors, scores, probs, cluster_weights, prior_stats

    def choose(self, bet_obj, details):
        _, prior_stats = self._load_model(bet_obj)
        _, _, probs, _, _ = self._event_features(bet_obj, details, prior_stats)
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
        _, prior_stats = self._load_model(bet_obj)
        details = []
        bet_outcomes = getattr(bet_obj, "outcomes", []) or []
        for o in bet_outcomes:
            details.append({
                "id": o.get("id", ""),
                "odds": float(o.get("odds", 1.01) or 1.01),
                "users_pct": float(o.get("percentage_users", 0.0) or 0.0),
                "points_pct": float(o.get("odds_percentage", 100.0 / max(1.01, float(o.get("odds", 1.01) or 1.01))) or 0.0),
                "total_users": float(o.get("total_users", 0.0) or 0.0),
                "total_points": float(o.get("total_points", 0.0) or 0.0),
            })
        _, scores, probs, _, _ = self._event_features(bet_obj, details, prior_stats)
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
        return amount, float(multiplier), "slrabbit99_cluster_ensemble_v1", "tier_slrabbit99_ensemble"
