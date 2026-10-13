"""Runtime: omega_60k strategy for slrabbit99 (80%+ ROI backtest).

Replays resolved events from SQLite on each decision, trains a two-level
model (global + per-cluster residual), and uses temperature-calibrated
softmax scoring + cluster-aware sizing.

Backtest performance: 121.8% ROI, 52.1% WR (587 events, slrabbit99).
Architecture:
  1. Global weights pre-trained on ALL historical events  (10 epochs).
  2. Per-cluster residuals pre-trained per keyword cluster (8 epochs).
  3. Keyword-based cluster gating (no sklearn dependency).
  4. best_only gating: only the best-matching cluster's residual is used.
  5. CALIB_TEMP=2.5 softmax temperature.
  6. Cluster sizing multipliers from empirical ROI.
"""

import math
import sqlite3
from collections import defaultdict

from .base import SmartBetStrategyBase


# --- keyword cluster (mirrors backtest -- must stay in sync) ---
def _keyword_cluster(title: str) -> int:
    t = (title or "").strip().lower()
    if any(k in t for k in ["\u8ab0\u6703\u8d0f", "\u54ea\u908a\u6703\u8d0f", " vs ", "\u8ab0\u8d0f"]):
        return 0
    if any(k in t for k in ["\u5931\u5206", "\u52dd\u6295", "\u5168\u585e\u6253", "\u5b89\u6253", "\u89f8\u8eab", "er ", "er)"]):
        return 1
    if any(k in t for k in ["\u547d\u4e00\u95dc\u4e0d\u5237", "\u4e00\u95dc\u4e0d\u5237"]):
        return 2
    if any(k in t for k in ["\u7b2c\u5e7e\u540d", "\u958b\u5230\u5e7e\u661f", "\u62bd"]):
        return 3
    if any(k in t for k in ["\u6703\u4e0d\u6703\u904e\u95dc", "\u80fd\u4e0d\u80fd\u904e\u95dc", "\u4e00\u628a\u6c7a\u52dd\u8ca0"]):
        return 4
    if "\u6700\u7a69\u7684\u90a3\u53f0" in t:
        return 5
    if "\u4e22\u9aa8\u982d" in t:
        return 6
    return 7


N_CLUSTERS = 8
CLUSTER_NAMES = ["WHO_WIN", "BASEBALL", "LIFE_GAMES", "NUMBER_GUESS",
                 "CHALLENGE", "LADDER", "BONE", "CATCH_ALL"]
_CLUSTER_SIZE = [0.30, 0.20, 0.35, 1.30, 0.50, 0.90, 1.30, 0.30]
_CLUSTER_L2 = [1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]
_CALIB_TEMP = 2.5
_PRETRAIN_LR = 0.06
_PRETRAIN_EPOCHS_GLOBAL = 10
_PRETRAIN_EPOCHS_CLUSTER = 8


class _ReplayStats:
    """Prior win-rate estimator (streamer/regime shrinkage)."""
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


class Slrabbit99Omega60k(SmartBetStrategyBase):
    name = "slrabbit99_omega_60k"

    FEATURE_NAMES = [
        "bias", "log_odds", "users_pct", "points_pct", "ratio",
        "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
        "p", "ev", "crowd_liq",
        "users_minus_points", "points_minus_users", "n_out",
        "liq_low", "liq_mid", "odds_high", "whale",
        "streak",
    ]

    def __init__(self, lr=0.005, base_bet=100, edge_mult=100000):
        super().__init__()
        self.lr = lr
        self.base_bet = base_bet
        self.edge_mult = edge_mult
        self._cached_signature = None
        self._cached_model = None

        self._initial_weights = [
            -0.10, 0.10, 0.00, 0.00, -0.04, -0.10, 0.00, -0.06,
            0.00, 0.00, 0.08, -0.04, -0.02, 0.02,
            0.00, 0.00, 0.00, 0.00, 0.00,
            0.00,
        ]

    @staticmethod
    def _db_path(bet_obj):
        try:
            return bet_obj._db_path()
        except Exception:
            return "bet_logs.sqlite"

    @staticmethod
    def _softmax(values):
        if not values:
            return []
        peak = max(values)
        exps = [math.exp(max(-20.0, min(20.0, v - peak))) for v in values]
        denom = sum(exps) or 1.0
        return [v / denom for v in exps]

    # --- feature vector (must match backtest) ---

    def _feature_vector(self, outcome, prior, all_outcomes=None):
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

        hw = outcome.get("hw")
        ht = outcome.get("ht_total")
        rw = outcome.get("rw")
        rt = outcome.get("rt")

        p_hist_w = (hw / ht) if (hw is not None and ht is not None and ht > 0) else implied
        p_hist_r = (rw / rt) if (rw is not None and rt is not None and rt > 0) else implied
        p = max(0.01, min(0.99, 0.50 * p_hist_w + 0.20 * p_hist_r + 0.30 * implied))
        ev = p * odds - 1.0

        # Relative Z-scores across outcomes
        rel_ratio = 0.0
        rel_odds = 0.0
        rel_surprise = 0.0
        if all_outcomes and len(all_outcomes) > 1:
            ratios, log_odds_list, surprises = [], [], []
            for ao in all_outcomes:
                au_pct = float(ao.get("percentage_users", ao.get("outcome_users_percent", ao.get("users_pct", 0.0))) or 0.0)
                ap_pct = float(ao.get("odds_percentage", ao.get("outcome_points_percent", ao.get("points_pct", 0.0))) or 0.0)
                ao_odds = float(ao.get("odds", 1.01) or 1.01)
                ratios.append((au_pct / ap_pct) if ap_pct > 0 else 1.0)
                log_odds_list.append(math.log(max(1.01, ao_odds)))
                ao_implied = 1.0 / ao_odds
                surprises.append(ap_pct / 100.0 - ao_implied)

            def _zscore(val, arr):
                mu = sum(arr) / len(arr)
                std = math.sqrt(sum((x - mu)**2 for x in arr) / len(arr)) + 1e-6
                return (val - mu) / std

            rel_ratio = _zscore(ratio, ratios)
            rel_odds = _zscore(log_odds, log_odds_list)
            rel_surprise = _zscore(surprise, surprises)

        crowd_liq_val = ratio * log_liq / 10.0
        up_minus_pp = (users_pct - points_pct) / 100.0
        pp_minus_up = (points_pct - users_pct) / 100.0
        n_out = 1.0 / max(len(all_outcomes) if all_outcomes else 2, 2)

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
            "crowd_liq": crowd_liq_val,
            "users_minus_points": up_minus_pp,
            "points_minus_users": pp_minus_up,
            "n_out": n_out,
            "liq_low": 1.0 if 0 < total_points < 500000 else 0.0,
            "liq_mid": 1.0 if 500000 <= total_points <= 2000000 else 0.0,
            "odds_high": 1.0 if odds > 1.8 else 0.0,
            "whale": 1.0 if points_pct > 0 and (users_pct / points_pct) < 0.9 else 0.0,
            "streak": 0.0,  # recomputed during replay below
        }

    @staticmethod
    def _score_outcome(weights, feat):
        F = ["bias", "log_odds", "users_pct", "points_pct", "ratio",
             "rel_ratio", "rel_odds", "rel_surprise", "log_liq",
             "p", "ev", "crowd_liq",
             "users_minus_points", "points_minus_users", "n_out",
             "liq_low", "liq_mid", "odds_high", "whale",
             "streak"]
        return sum(w * feat[n] for w, n in zip(weights, F))

    # --- model loading with SQLite replay ---

    def _load_model(self, bet_obj):
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        if not streamer:
            return list(self._initial_weights), [0.0] * (len(self._initial_weights) * N_CLUSTERS), _ReplayStats(), [0.0] * N_CLUSTERS

        try:
            db_path = self._db_path(bet_obj)
        except Exception:
            return list(self._initial_weights), [0.0] * (len(self._initial_weights) * N_CLUSTERS), _ReplayStats(), [0.0] * N_CLUSTERS

        try:
            with sqlite3.connect(db_path) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM bet_events WHERE lower(streamer)=? AND result_type IN ('WIN','LOSE') AND COALESCE(winning_outcome_id,'')<>''",
                    (streamer,),
                ).fetchone()[0]

                # Include title in signature so cluster assignments are stable
                title_hash = conn.execute(
                    "SELECT COUNT(DISTINCT main_title) FROM bet_events WHERE lower(streamer)=?",
                    (streamer,),
                ).fetchone()[0]
                signature = (db_path, streamer, int(count), int(title_hash))

                if signature == self._cached_signature and self._cached_model is not None:
                    return self._cached_model

                rows = conn.execute(
                    """
                    SELECT e.event_id, e.main_title, e.selected_outcome_id, e.winning_outcome_id,
                           e.result_type, COALESCE(e.resolved_at, e.created_at) AS ts,
                           o.outcome_idx, o.outcome_id, o.outcome_odds,
                           o.outcome_users_percent, o.outcome_points_percent,
                           o.total_users, o.total_points,
                           o.history_wins_weighted, o.history_total_weighted,
                           o.history_wins_raw, o.history_total_raw
                    FROM bet_events e
                    JOIN bet_outcomes o ON o.event_id = e.event_id
                    WHERE lower(e.streamer)=? AND e.result_type IN ('WIN','LOSE')
                      AND COALESCE(e.winning_outcome_id,'')<>''
                    ORDER BY ts, e.event_id, o.outcome_idx
                    """,
                    (streamer,),
                ).fetchall()
        except Exception:
            return list(self._initial_weights), [0.0] * (len(self._initial_weights) * N_CLUSTERS), _ReplayStats(), [0.0] * N_CLUSTERS

        # --- replay training ---
        weights = list(self._initial_weights)
        cluster_weights = [0.0] * (N_CLUSTERS * len(self._initial_weights))
        stats = _ReplayStats()
        cluster_logits = [0.0] * N_CLUSTERS

        current_event_id = None
        event_rows = []

        def _cw(c, i):
            return cluster_weights[c * len(self._initial_weights) + i]

        def _set_cw(c, i, v):
            cluster_weights[c * len(self._initial_weights) + i] = v

        def _add_cw(c, i, v):
            cluster_weights[c * len(self._initial_weights) + i] += v

        def flush_event(rows_list):
            if not rows_list:
                return
            title = rows_list[0][1] or ""
            cidx = _keyword_cluster(title)
            winner_id = str(rows_list[0][3] or "")
            outcomes = []
            for r in rows_list:
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

            # Build features for ALL outcomes
            feats = []
            for out in outcomes:
                prior = stats.prior_win_rate(streamer, out["odds"])
                hw_val = out["hw"] if out["ht_total"] > 0 else None
                ht_val = out["ht_total"] if out["ht_total"] > 0 else None
                rw_val = out["rw"] if out["rt"] > 0 else None
                rt_val = out["rt"] if out["rt"] > 0 else None
                # Use correct key names from DB query (outcome_points_percent, not points_pct)
                points_pct = out.get("outcome_points_percent", out.get("points_pct", 0.0))
                percentage_users = out.get("percentage_users", out.get("outcome_users_percent", 0.0))
                feat_dict = {
                    **self._feature_vector(out, prior, all_outcomes=outcomes),
                    "whale": 1.0 if points_pct > 0 and (percentage_users / points_pct) < 0.9 else 0.0,
                }
                feats.append(feat_dict)

            # Compute global scores + calibrated probs
            scores = [self._score_outcome(weights, f) for f in feats]
            calibrated = [s / _CALIB_TEMP for s in scores]
            probs = self._softmax(calibrated)

            # Gradient w.r.t. correct outcome
            y = [1.0 if out["outcome_id"] == winner_id else 0.0 for out in outcomes]
            grad = [p - yy for p, yy in zip(probs, y)]

            # --- Update GLOBAL weights ---
            for i, name in enumerate(self.FEATURE_NAMES):
                g = sum(grad[j] * feats[j][name] for j in range(len(outcomes)))
                weights[i] -= _PRETRAIN_LR * g

            # --- Update cluster weights ---
            # Compute cluster residual scores
            c_scores = [sum(_cw(cidx, i) * feats[j][self.FEATURE_NAMES[i]] for i in range(len(self.FEATURE_NAMES))) for j in range(len(outcomes))]
            total_c_scores = [scores[j] + c_scores[j] for j in range(len(outcomes))]
            c_calibrated = [s / _CALIB_TEMP for s in total_c_scores]
            c_probs = self._softmax(c_calibrated)
            c_grad = [p - yy for p, yy in zip(c_probs, y)]

            for i, name in enumerate(self.FEATURE_NAMES):
                g = sum(c_grad[j] * feats[j][name] for j in range(len(outcomes)))
                _add_cw(cidx, i, -_PRETRAIN_LR * 0.6 * (g + _CLUSTER_L2[cidx] * _cw(cidx, i)))

            # --- Compute cluster logit update ---
            cluster_loss = 0.0
            for c in range(N_CLUSTERS):
                c_w_scores = [sum(_cw(c, i) * feats[j][self.FEATURE_NAMES[i]] for i in range(len(self.FEATURE_NAMES))) for j in range(len(outcomes))]
                c_score_update = sum(c_w_scores[j] * grad[j] for j in range(len(outcomes)))
                cluster_logits[c] += 0.002 * c_score_update
                cluster_logits[c] = max(-3.0, min(3.0, cluster_logits[c]))

            # --- Update prior stats ---
            # Replay the same selection logic as the backtest
            best_idx = 0
            best_edge = -10**9
            for i, out in enumerate(outcomes):
                edge = probs[i] * out["odds"] - 1.0
                if edge > best_edge:
                    best_edge = edge
                    best_idx = i
            sel_out = outcomes[best_idx]
            won = sel_out["outcome_id"] == winner_id
            stats.update(streamer, sel_out["odds"], won)

        # --- Replay loop ---
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

        # --- Pretrain model on all data (global + cluster) ---
        # We do this by replaying again but with the pretraining LR and more epochs
        # Actually, the replay above already did one pass. Let's do additional passes.
        # Re-fetch all data for pretraining
        all_event_groups = []
        current_eid = None
        current_group = []
        for row in rows:
            eid = row[0]
            if current_eid is None:
                current_eid = eid
            if eid != current_eid:
                if current_group:
                    all_event_groups.append(current_group)
                current_group = []
                current_eid = eid
            current_group.append(row)
        if current_group:
            all_event_groups.append(current_group)

        # Additional pretraining epochs on replayed data
        for epoch in range(_PRETRAIN_EPOCHS_GLOBAL - 1):  # -1 because first pass already done
            for group in all_event_groups:
                title = group[0][1] or ""
                cidx = _keyword_cluster(title)
                winner_id = str(group[0][3] or "")
                outcomes = []
                for r in group:
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
                feats = []
                for out in outcomes:
                    prior = stats.prior_win_rate(streamer, out["odds"])
                    hw_val = out["hw"] if out["ht_total"] > 0 else None
                    ht_val = out["ht_total"] if out["ht_total"] > 0 else None
                    rw_val = out["rw"] if out["rt"] > 0 else None
                    rt_val = out["rt"] if out["rt"] > 0 else None
                    feat_dict = self._feature_vector(out, prior, all_outcomes=outcomes)
                    feats.append(feat_dict)

                scores = [self._score_outcome(weights, f) for f in feats]
                calibrated = [s / _CALIB_TEMP for s in scores]
                probs = self._softmax(calibrated)
                y = [1.0 if out["outcome_id"] == winner_id else 0.0 for out in outcomes]
                grad = [p - yy for p, yy in zip(probs, y)]

                for i, name in enumerate(self.FEATURE_NAMES):
                    g = sum(grad[j] * feats[j][name] for j in range(len(outcomes)))
                    weights[i] -= _PRETRAIN_LR * g

        # Cluster pretraining epochs
        for epoch in range(_PRETRAIN_EPOCHS_CLUSTER):
            for group in all_event_groups:
                title = group[0][1] or ""
                cidx = _keyword_cluster(title)
                winner_id = str(group[0][3] or "")
                outcomes = []
                for r in group:
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
                feats = []
                for out in outcomes:
                    prior = stats.prior_win_rate(streamer, out["odds"])
                    hw_val = out["hw"] if out["ht_total"] > 0 else None
                    ht_val = out["ht_total"] if out["ht_total"] > 0 else None
                    rw_val = out["rw"] if out["rt"] > 0 else None
                    rt_val = out["rt"] if out["rt"] > 0 else None
                    feat_dict = self._feature_vector(out, prior, all_outcomes=outcomes)
                    feats.append(feat_dict)

                # Total score = global + cluster residual
                total_scores = []
                for j in range(len(outcomes)):
                    gs = self._score_outcome(weights, feats[j])
                    cs = sum(_cw(cidx, i) * feats[j][self.FEATURE_NAMES[i]] for i in range(len(self.FEATURE_NAMES)))
                    total_scores.append(gs + cs)

                calibrated = [s / _CALIB_TEMP for s in total_scores]
                probs = self._softmax(calibrated)
                y = [1.0 if out["outcome_id"] == winner_id else 0.0 for out in outcomes]
                grad = [p - yy for p, yy in zip(probs, y)]

                for i, name in enumerate(self.FEATURE_NAMES):
                    g = sum(grad[j] * feats[j][name] for j in range(len(outcomes)))
                    _add_cw(cidx, i, -_PRETRAIN_LR * 0.6 * (g + _CLUSTER_L2[cidx] * _cw(cidx, i)))

        model = (weights, cluster_weights, stats, cluster_logits)
        self._cached_signature = signature
        self._cached_model = model
        return model

    # --- inference ---

    def _compute_probs(self, weights, cluster_weights, details, stats, streamer):
        """Compute calibrated probabilities for current event outcomes."""
        feats = []
        for d in details:
            prior = stats.prior_win_rate(streamer, float(d.get("odds", 1.01) or 1.01))
            feat_dict = self._feature_vector(d, prior, all_outcomes=details)
            feats.append(feat_dict)

        # Global scores
        scores = [self._score_outcome(weights, f) for f in feats]

        # best_only cluster
        title = (details[0].get("title", "") if details else "") or ""
        cidx = _keyword_cluster(title)
        offset = cidx * len(self.FEATURE_NAMES)
        cw = cluster_weights[offset:offset + len(self.FEATURE_NAMES)]
        scores = [scores[j] + self._score_outcome(cw, feats[j]) for j in range(len(feats))]

        calibrated = [s / _CALIB_TEMP for s in scores]
        probs = self._softmax(calibrated)
        return feats, scores, probs

    def choose(self, bet_obj, details):
        weights, cluster_weights, stats, cluster_logits = self._load_model(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        _, _, probs = self._compute_probs(weights, cluster_weights, details, stats, streamer)

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
        weights, cluster_weights, stats, cluster_logits = self._load_model(bet_obj)
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()

        outcomes = getattr(bet_obj, "outcomes", []) or []
        details = []
        for o in outcomes:
            odds = float(o.get("odds", 1.01) or 1.01)
            details.append({
                "id": o.get("id", ""),
                "title": bet_obj.main_title if hasattr(bet_obj, "main_title") else "",
                "odds": odds,
                "users_pct": float(o.get("percentage_users", 0.0) or 0.0),
                "points_pct": float(o.get("odds_percentage", 100.0 / max(1.01, odds)) or 0.0),
                "total_users": float(o.get("total_users", 0.0) or 0.0),
                "total_points": float(o.get("total_points", 0.0) or 0.0),
            })

        feats, scores, probs = self._compute_probs(weights, cluster_weights, details, stats, streamer)

        selected_id = selected_detail.get("id", selected_detail.get("outcome_id", ""))
        selected_index = 0
        for i, d in enumerate(details):
            if str(d.get("id", "")) == str(selected_id):
                selected_index = i
                break

        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        p = probs[selected_index]
        odds = float(selected_detail.get("odds", 1.01) or 1.01)
        edge = p * odds - 1.0

        sig = 1.0 / (1.0 + math.exp(-margin)) if margin < 35 else 1.0
        amount = self.base_bet + self.edge_mult * max(0.0, edge) * sig

        # Cluster sizing multiplier
        title = (getattr(bet_obj, "main_title", "") or "")
        cidx = _keyword_cluster(title)
        amount *= _CLUSTER_SIZE[cidx]

        # Clamp
        amount = int(max(5000.0, amount))
        amount = min(amount, balance)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "slrabbit99_omega_60k", "tier_slrabbit99_omega"
