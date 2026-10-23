import math
import sqlite3
from collections import defaultdict
from .base import SmartBetStrategyBase


class Roger9527CustomV2(SmartBetStrategyBase):
    """
    Runtime version of roger9527_custom_v2 backtest strategy.
    Pure online learning, per-title bandit, no look-ahead features.
    Rebuilds model from SQLite on each decision.
    """

    name = "roger9527_custom_v2"

    GOOD_TITLE_KEYWORDS = [
        "兩張", "怎麼說", "先死", "前25", "出國", "60%",
        "鈴珠", "結果", "00:05", "00:45", "關盤", "退",
    ]
    BAD_TITLE_KEYWORDS = [
        "這把會", "瘸瘸先", "這把輸贏?", "北投超威", "北投超萎",
        "底迪會被抓到", "這輪", "這把先拿到", "先?", "哪個先",
        "真真正正最後一把", "10:00時瘸瘸", "藍寶堅尼", "夥伴獵",
    ]

    def __init__(self):
        self.lr = 0.05
        self.reg = 1e-4
        self.lr_signal = 0.1
        self.w = [0.0] * 6
        self.ev_weight = 0.5
        self.whale_weight = 0.3
        self.contrarian_weight = 0.0

        # Per-title bandits (loaded from DB each decision)
        self.title_bandits = defaultdict(lambda: defaultdict(lambda: {"n": 0, "avg": 0.0}))

    def _title_key(self, title: str) -> str:
        for kw in self.GOOD_TITLE_KEYWORDS:
            if kw in title:
                return f"GOOD:{kw}"
        for kw in self.BAD_TITLE_KEYWORDS:
            if kw in title:
                return f"BAD:{kw}"
        return "NEUTRAL"

    def _arm(self, odds):
        if odds <= 1.8:
            return "A"
        if odds <= 2.2:
            return "B"
        if odds <= 3.0:
            return "C"
        if odds <= 5.0:
            return "D"
        return "E"

    def _load_bandits_from_db(self, bet_obj):
        """Load per-title bandit stats from bet_logs.sqlite"""
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        if not streamer:
            return

        try:
            db_path = bet_obj._db_path()
        except Exception:
            return

        sql = """
        SELECT
            e.main_title,
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
        GROUP BY e.main_title, arm
        """

        try:
            with sqlite3.connect(db_path) as conn:
                for title, arm, n, avg_pnl in conn.execute(sql, (streamer,)).fetchall():
                    tkey = self._title_key(title)
                    if arm in self.title_bandits[tkey]:
                        self.title_bandits[tkey][arm]["n"] = int(n or 0)
                        self.title_bandits[tkey][arm]["avg"] = float(avg_pnl or 0.0)
        except Exception:
            pass

    def _load_model_from_db(self, bet_obj):
        """Train online logistic regression from historical resolved events"""
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        if not streamer:
            return

        try:
            db_path = bet_obj._db_path()
        except Exception:
            return

        sql = """
        SELECT
            o.outcome_odds,
            o.outcome_users_percent,
            o.outcome_points_percent,
            o.total_points,
            e.winning_outcome_id,
            e.selected_outcome_id
        FROM bet_events e
        JOIN bet_outcomes o
          ON o.event_id = e.event_id
         AND o.outcome_id = e.selected_outcome_id
        WHERE lower(e.streamer) = ?
          AND e.result_type IN ('WIN', 'LOSE')
          AND COALESCE(e.winning_outcome_id, '') <> ''
        ORDER BY e.created_at
        """

        try:
            with sqlite3.connect(db_path) as conn:
                for odds, users_pct, points_pct, total_points, winner_id, selected_id in conn.execute(sql, (streamer,)).fetchall():
                    won = (selected_id == winner_id)
                    y = 1.0 if won else 0.0  # train on BOTH classes — critical fix

                    prior = 0.5
                    feat = self._clean_features(
                        odds, users_pct, points_pct, total_points, prior
                    )
                    x = [1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]]
                    p = max(0.02, min(0.98, self._pred(feat)))
                    g = p - y
                    for i in range(len(self.w)):
                        self.w[i] -= self.lr * (g * x[i] + self.reg * self.w[i])
        except Exception:
            pass

    def _clean_features(self, odds, users_pct, points_pct, total_points, prior_p):
        ratio = (users_pct / points_pct) if points_pct > 0 else 1.0
        log_odds = math.log(max(1.01, odds))
        implied_p = 1.0 / odds
        p = max(0.01, min(0.99, 0.3 * prior_p + 0.7 * implied_p))
        ev = p * odds - 1.0
        conf = min(1.0, (total_points / 1_000_000) * 0.5 + 0.3)
        whale_signal = -ratio if ratio < 0.9 else 0.0

        return {
            "ratio": ratio,
            "log_odds": log_odds,
            "p": p,
            "ev": ev,
            "conf": conf,
            "whale": whale_signal,
        }

    def _pred(self, feat):
        z = sum(a * b for a, b in zip(self.w, [
            1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]
        ]))
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _score_outcome(self, odds, users_pct, points_pct, total_points, prior_p, title_key):
        feat = self._clean_features(odds, users_pct, points_pct, total_points, prior_p)
        p_model = max(0.02, min(0.98, self._pred(feat)))

        arm = self._arm(odds)
        bandit = self.title_bandits[title_key]
        bandit_score = bandit[arm]["avg"] if bandit[arm]["n"] > 2 else 0.0

        ev_signal = feat["ev"] * self.ev_weight
        whale_signal = feat["whale"] * self.whale_weight

        score = p_model * odds - 1.0 + 0.1 * (ev_signal + whale_signal) + 0.2 * bandit_score
        return score, p_model, feat, arm

    def choose(self, bet_obj, details):
        # Load model and bandits from DB
        self._load_model_from_db(bet_obj)
        self._load_bandits_from_db(bet_obj)

        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        title = ""
        for d in details:
            if d.get("main_title"):
                title = d["main_title"]
                break
        title_key = self._title_key(title)

        prior_p = 0.5
        best_index = 0
        best_score = -1e9

        for i, d in enumerate(details):
            odds = d.get("odds", 1.0)
            users_pct = d.get("users_percent", 0) * 100
            points_pct = d.get("points_percent", 0) * 100
            total_points = d.get("total_points", 0)

            score, p_model, feat, arm = self._score_outcome(
                odds, users_pct, points_pct, total_points, prior_p, title_key
            )

            if score > best_score:
                best_score = score
                best_index = i
                self._last_feat = feat
                self._last_arm = arm
                self._last_p_model = p_model
                self._last_odds = odds

        return best_index

    def size(self, bet_obj, balance, selected_detail):
        odds = selected_detail.get("odds", 1.0)
        ev = selected_detail.get("ev", 0.0)

        feat = getattr(self, "_last_feat", None)
        arm = getattr(self, "_last_arm", "C")
        p_model = getattr(self, "_last_p_model", 0.5)

        if feat is None:
            feat = self._clean_features(odds, selected_detail.get("users_percent", 0)*100,
                                         selected_detail.get("points_percent", 0)*100,
                                         selected_detail.get("total_points", 0), 0.5)
            arm = self._arm(odds)

        # Bandit confidence
        streamer = (getattr(bet_obj, "streamer_name", "") or "").strip().lower()
        title = ""
        for d in bet_obj.details if hasattr(bet_obj, "details") else []:
            if d.get("main_title"):
                title = d["main_title"]
                break
        title_key = self._title_key(title)
        bandit = self.title_bandits[title_key]
        bandit_conf = min(1.0, bandit[arm]["n"] / 10.0)

        conf = feat["conf"] * (0.5 + 0.5 * bandit_conf)
        base = 20000 + 60000 * conf
        ev_amount = 100000 * max(0, ev)

        if title_key.startswith("GOOD"):
            title_mult = 1.3
        elif title_key.startswith("BAD"):
            title_mult = 0.5
        else:
            title_mult = 1.0

        amount = int((base + ev_amount) * title_mult)

        if odds > 4.0:
            amount = int(amount * 0.7)
        elif odds > 2.5:
            amount = int(amount * 0.85)

        # Apply bounds
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), f"roger9527_v2_{arm}", "tier_custom"