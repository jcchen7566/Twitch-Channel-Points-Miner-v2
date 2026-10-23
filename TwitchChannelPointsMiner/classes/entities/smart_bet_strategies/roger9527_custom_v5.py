import math
import sqlite3
from collections import defaultdict
from .base import SmartBetStrategyBase


class Roger9527CustomV5(SmartBetStrategyBase):
    """
    Runtime version of roger9527_custom_v5 backtest strategy.
    Pure online learning, per-title bandit, EV-based scoring, max odds filter.
    Rebuilds model from SQLite on each decision (wins AND losses).
    """

    name = "roger9527_custom_v5"

    GAMING_BOSS_KEYWORDS = [
        "套敏", "兩張", "60%", "死", "次過", "前能過", "前會過",
        "鈴珠", "獵人", "災厄", "影子", "紅狼", "女王", "龍",
        "將軍", "拉塔恩", "腐敗", "樹靈", "墜星", "艾絲緹",
        "大肥哥", "神皮", "貴族", "王", "惡兆", "接肢", "巴洛古",
        "礦坑", "boss", "滿月", "火龍", "土龍", "一次過",
    ]
    GENERIC_WINLOSS_KEYWORDS = [
        "這把會", "這把輸贏", "本場勝負", "這把勝負", "先?",
        "哪個先", "真真正正最後一把", "這ˇ把怎麼說", "這把結果",
        "出國", "夥伴獵", "藍寶堅尼", "北投超威", "北投超萎",
        "底迪", "鈴珠獵人 三次過", "真高手羅傑", "瘸瘸", "10:00時瘸瘸",
    ]

    MAX_ODDS_BET = 8.0

    def __init__(self):
        self.lr = 0.03
        self.reg = 1e-4
        self.lr_signal = 0.1
        self.w = [0.0] * 6
        self.ev_weight = 0.5
        self.whale_weight = 0.3
        self.title_bandits = defaultdict(lambda: defaultdict(lambda: {"n": 0, "avg": 0.0}))
        self.title_total = defaultdict(int)
        self.title_wins = defaultdict(int)

    def _cluster_key(self, title: str) -> str:
        for kw in self.GAMING_BOSS_KEYWORDS:
            if kw in title:
                return "GAMING_BOSS"
        for kw in self.GENERIC_WINLOSS_KEYWORDS:
            if kw in title:
                return "GENERIC_WINLOSS"
        return "OTHER"

    def _arm(self, odds):
        if odds <= 1.8: return "A"
        if odds <= 2.2: return "B"
        if odds <= 3.0: return "C"
        if odds <= 5.0: return "D"
        return "E"

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
            "odds": odds,
        }

    def _pred(self, feat):
        z = sum(a * b for a, b in zip(self.w, [
            1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]
        ]))
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _load_model_from_db(self, bet_obj):
        """Train online logistic regression from ALL resolved events (wins AND losses)."""
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
                rows = conn.execute(sql, (streamer,)).fetchall()

            # Reset weights each decision: single source of truth = DB history
            self.w = [0.0] * 6
            for odds, users_pct, points_pct, total_points, winner_id, selected_id in rows:
                won = (selected_id == winner_id)
                y = 1.0 if won else 0.0  # BOTH classes — critical fix
                prior = 0.5
                feat = self._clean_features(odds, users_pct, points_pct, total_points, prior)
                x = [1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]]
                p = max(0.02, min(0.98, self._pred(feat)))
                g = p - y
                for i in range(len(self.w)):
                    self.w[i] -= self.lr * (g * x[i] + self.reg * self.w[i])
        except Exception:
            pass

    def _load_bandits_from_db(self, bet_obj):
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

        self.title_bandits = defaultdict(lambda: defaultdict(lambda: {"n": 0, "avg": 0.0}))
        try:
            with sqlite3.connect(db_path) as conn:
                for title, arm, n, avg_pnl in conn.execute(sql, (streamer,)).fetchall():
                    if arm in self.title_bandits[title]:
                        self.title_bandits[title][arm]["n"] = int(n or 0)
                        self.title_bandits[title][arm]["avg"] = float(avg_pnl or 0.0)
        except Exception:
            pass

        sql2 = """
        SELECT
            e.main_title,
            COUNT(*) AS total,
            SUM(CASE WHEN e.selected_outcome_id = e.winning_outcome_id THEN 1 ELSE 0 END) AS wins
        FROM bet_events e
        JOIN bet_outcomes o
          ON o.event_id = e.event_id
         AND o.outcome_id = e.selected_outcome_id
        WHERE lower(e.streamer) = ?
          AND e.result_type IN ('WIN', 'LOSE')
          AND COALESCE(e.winning_outcome_id, '') <> ''
        GROUP BY e.main_title
        """
        self.title_total = defaultdict(int)
        self.title_wins = defaultdict(int)
        try:
            with sqlite3.connect(db_path) as conn:
                for title, total, wins in conn.execute(sql2, (streamer,)).fetchall():
                    self.title_total[title] = int(total or 0)
                    self.title_wins[title] = int(wins or 0)
        except Exception:
            pass

    def _title_multiplier(self, title: str, cluster: str) -> float:
        total = self.title_total.get(title, 0)
        if total >= 10:
            wr = self.title_wins.get(title, 0) / total
            if cluster == "GAMING_BOSS" and wr >= 0.7:
                return 1.3
            elif cluster == "GENERIC_WINLOSS" and wr >= 0.65:
                return 1.1
            elif wr < 0.4:
                return 0.6
        elif total >= 5:
            wr = self.title_wins.get(title, 0) / total
            if cluster == "GAMING_BOSS" and wr >= 0.75:
                return 1.15
            elif wr < 0.35:
                return 0.7
        if cluster == "GAMING_BOSS":
            return 1.0
        elif cluster == "GENERIC_WINLOSS":
            return 0.8
        return 1.0

    def choose(self, bet_obj, details):
        self._load_model_from_db(bet_obj)
        self._load_bandits_from_db(bet_obj)

        title = ""
        for d in details:
            if d.get("main_title"):
                title = d["main_title"]
                break
        if not title:
            title = getattr(bet_obj, "title", "") or ""
        cluster = self._cluster_key(title)

        prior_p = 0.5
        best_index = 0
        best_score = -1e9

        for i, d in enumerate(details):
            odds = float(d.get("odds", 1.0) or 1.0)
            if odds <= 1.0:
                odds = 1.01
            if odds > self.MAX_ODDS_BET:
                continue

            # Real crowd data from bet_obj.outcomes (details dict lacks these keys)
            try:
                out = bet_obj.outcomes[i]
                users_pct = float(out.get("percentage_users", 0) or 0)
                total_points = float(getattr(bet_obj, "total_points", 0) or 0)
                pts = float(out.get("total_points", 0) or 0)
                points_pct = (100.0 * pts / total_points) if total_points > 0 else 0.0
            except Exception:
                users_pct, points_pct, total_points = 0.0, 0.0, 0.0

            feat = self._clean_features(odds, users_pct, points_pct, total_points, prior_p)
            p_model = max(0.02, min(0.98, self._pred(feat)))

            arm = self._arm(odds)
            bandit = self.title_bandits[title]
            bandit_score = bandit[arm]["avg"] if bandit[arm]["n"] > 2 else 0.0

            ev_signal = feat["ev"] * self.ev_weight
            whale_signal = feat["whale"] * self.whale_weight

            score = feat["ev"] + 0.1 * (ev_signal + whale_signal) + 0.2 * bandit_score

            if score > best_score:
                best_score = score
                best_index = i
                self._last_feat = feat
                self._last_arm = arm
                self._last_odds = odds
                self._last_title = title
                self._last_cluster = cluster

        return best_index

    def size(self, bet_obj, balance, selected_detail):
        odds = float(selected_detail.get("odds", 1.0) or 1.0)
        if odds > self.MAX_ODDS_BET:
            odds = self.MAX_ODDS_BET

        feat = getattr(self, "_last_feat", None)
        arm = getattr(self, "_last_arm", "C")
        title = getattr(self, "_last_title", "")
        cluster = getattr(self, "_last_cluster", "OTHER")

        if feat is None:
            feat = self._clean_features(odds, 0.0, 0.0, 0.0, 0.5)
            arm = self._arm(odds)

        bandit = self.title_bandits[title]
        bandit_conf = min(1.0, bandit[arm]["n"] / 10.0)
        title_conf = min(1.0, self.title_total.get(title, 0) / 20.0)

        conf = feat["conf"] * (0.5 + 0.3 * bandit_conf + 0.2 * title_conf)
        base = 20000 + 50000 * conf
        p_model = max(0.02, min(0.98, self._pred(feat)))
        ev = p_model * odds - 1.0
        ev_amount = 80000 * max(0, ev)

        title_mult = self._title_multiplier(title, cluster)
        amount = int((base + ev_amount) * title_mult)

        # Hard cap on cold titles
        if self.title_total.get(title, 0) < 5:
            amount = min(amount, 40000)
        elif self.title_total.get(title, 0) < 10:
            amount = min(amount, 60000)

        if odds > 4.0:
            amount = int(amount * 0.7)
        elif odds > 2.5:
            amount = int(amount * 0.85)

        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), f"roger9527_v5_{arm}", "tier_custom"