import math
from collections import defaultdict
from ..core.features import clamp, liquidity_bucket, odds_regime
from ..core.strategy_base import StrategyBase, OnlineStats


class Roger9527CustomV4(StrategyBase):
    """
    roger9527 custom v4 - Conservative cold-start, per-title bandit with min samples
    - Only boost titles with proven track record (min 5 events)
    - Cap bet size on new/unproven titles
    - Pure online, no look-ahead features
    """
    name = "roger9527_custom_v4"

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

    def __init__(self):
        self.cluster_weights = defaultdict(lambda: [0.0] * 6)
        self.lr = 0.05
        self.reg = 1e-4

        # Per-title bandit
        self.title_bandits = defaultdict(lambda: defaultdict(lambda: {"n": 0, "avg": 0.0}))
        self.title_total_events = defaultdict(int)

        # Cluster signal weights
        self.cluster_ev_weight = defaultdict(lambda: 0.5)
        self.cluster_whale_weight = defaultdict(lambda: 0.3)
        self.lr_signal = 0.1

        # Title performance tracking
        self.title_wins = defaultdict(int)
        self.title_losses = defaultdict(int)

    def _cluster_key(self, title: str) -> str:
        for kw in self.GAMING_BOSS_KEYWORDS:
            if kw in title:
                return "GAMING_BOSS"
        for kw in self.GENERIC_WINLOSS_KEYWORDS:
            if kw in title:
                return "GENERIC_WINLOSS"
        return "OTHER"

    def _clean_features(self, o, prior_p, event_outcomes):
        ratio = (o.users_pct / o.points_pct) if o.points_pct > 0 else 1.0
        log_odds = math.log(max(1.01, o.odds))
        implied_p = 1.0 / o.odds
        p = clamp(0.3 * prior_p + 0.7 * implied_p, 0.01, 0.99)
        ev = p * o.odds - 1.0

        rel_ratio = 0.0
        if event_outcomes and len(event_outcomes) > 1:
            all_ratios = [(x.users_pct / x.points_pct) if x.points_pct > 0 else 1.0 for x in event_outcomes]
            mu_r = sum(all_ratios) / len(all_ratios)
            std_r = math.sqrt(sum((x - mu_r)**2 for x in all_ratios) / len(all_ratios)) + 1e-6
            rel_ratio = (ratio - mu_r) / std_r

        conf = min(1.0, (o.total_points / 1_000_000) * 0.5 + 0.3)
        whale_signal = -ratio if ratio < 0.9 else 0.0

        return {
            "ratio": ratio,
            "rel_ratio": rel_ratio,
            "log_odds": log_odds,
            "p": p,
            "ev": ev,
            "conf": conf,
            "whale": whale_signal,
        }

    def _pred(self, cluster, feat):
        w = self.cluster_weights[cluster]
        z = sum(a * b for a, b in zip(w, [
            1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]
        ]))
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _arm(self, odds):
        if odds <= 1.8: return "A"
        if odds <= 2.2: return "B"
        if odds <= 3.0: return "C"
        if odds <= 5.0: return "D"
        return "E"

    def _title_multiplier(self, title: str, cluster: str) -> float:
        """Conservative multiplier: only boost proven titles"""
        total = self.title_total_events[title]
        if total >= 10:
            wins = self.title_wins[title]
            wr = wins / total
            if cluster == "GAMING_BOSS" and wr >= 0.7:
                return 1.3
            elif cluster == "GENERIC_WINLOSS" and wr >= 0.65:
                return 1.1
            elif wr < 0.4:
                return 0.6
        elif total >= 5:
            wins = self.title_wins[title]
            wr = wins / total
            if cluster == "GAMING_BOSS" and wr >= 0.75:
                return 1.15
            elif wr < 0.35:
                return 0.7
        # Cold start or poor performance
        if cluster == "GAMING_BOSS":
            return 1.0  # Neutral, not boosted
        elif cluster == "GENERIC_WINLOSS":
            return 0.8  # Slightly conservative
        return 1.0

    def pick(self, event, state):
        cluster = self._cluster_key(event.main_title)
        bandit = self.title_bandits[event.main_title]
        weights = self.cluster_weights[cluster]
        ev_w = self.cluster_ev_weight[cluster]
        whale_w = self.cluster_whale_weight[cluster]

        best = None
        best_score = -1e9

        for o in event.outcomes:
            prior = state.prior_win_rate(event.streamer, o.odds)
            feat = self._clean_features(o, prior, event.outcomes)
            p_model = max(0.02, min(0.98, self._pred(cluster, feat)))

            arm = self._arm(o.odds)
            bandit_score = bandit[arm]["avg"] if bandit[arm]["n"] > 2 else 0.0

            ev_signal = feat["ev"] * ev_w
            whale_signal = feat["whale"] * whale_w

            score = p_model * o.odds - 1.0 + 0.1 * (ev_signal + whale_signal) + 0.2 * bandit_score

            if score > best_score:
                best_score = score
                best = o
                best_feat = feat
                best_arm = arm

        self._last_cluster = cluster
        self._last_feat = best_feat
        self._last_arm = best_arm
        return best

    def size(self, event, selected, state):
        cluster = getattr(self, "_last_cluster", "OTHER")
        feat = getattr(self, "_last_feat", None)
        arm = getattr(self, "_last_arm", "C")
        p_model = max(0.02, min(0.98, self._pred(cluster, feat))) if feat else 0.5

        if feat is None:
            prior = state.prior_win_rate(event.streamer, selected.odds)
            feat = self._clean_features(selected, prior, event.outcomes)
            arm = self._arm(selected.odds)

        bandit = self.title_bandits[event.main_title]
        bandit_conf = min(1.0, bandit[arm]["n"] / 10.0)
        title_conf = min(1.0, self.title_total_events[event.main_title] / 20.0)

        conf = feat["conf"] * (0.5 + 0.3 * bandit_conf + 0.2 * title_conf)
        base = 20000 + 50000 * conf
        ev = p_model * selected.odds - 1.0
        ev_amount = 80000 * max(0, ev)

        title_mult = self._title_multiplier(event.main_title, cluster)
        amount = int((base + ev_amount) * title_mult)

        # Hard cap on cold titles
        if self.title_total_events[event.main_title] < 5:
            amount = min(amount, 50000)
        elif self.title_total_events[event.main_title] < 10:
            amount = min(amount, 80000)

        if selected.odds > 4.0:
            amount = int(amount * 0.7)
        elif selected.odds > 2.5:
            amount = int(amount * 0.85)

        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        cluster = self._cluster_key(event.main_title)
        bandit = self.title_bandits[event.main_title]
        arm = getattr(self, "_last_arm", self._arm(selected.odds))

        # Update title bandit
        b = bandit[arm]
        b["n"] += 1
        b["avg"] += (pnl - b["avg"]) / b["n"]

        # Update title stats
        self.title_total_events[event.main_title] += 1
        if won:
            self.title_wins[event.main_title] += 1
        else:
            self.title_losses[event.main_title] += 1

        # Update cluster logistic regression
        prior = state.prior_win_rate(event.streamer, selected.odds)
        feat = self._clean_features(selected, prior, event.outcomes)
        w = self.cluster_weights[cluster]
        x = [1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]]
        p = max(0.02, min(0.98, self._pred(cluster, feat)))
        y = 1.0 if won else 0.0
        g = p - y
        for i in range(len(w)):
            w[i] -= self.lr * (g * x[i] + self.reg * w[i])

        # Update cluster signal weights
        winner = next((o for o in event.outcomes if o.outcome_id == event.winner_outcome_id), None)
        if winner:
            w_prior = state.prior_win_rate(event.streamer, winner.odds)
            w_feat = self._clean_features(winner, w_prior, event.outcomes)
            self.cluster_ev_weight[cluster] += self.lr_signal * (w_feat["ev"] - self.cluster_ev_weight[cluster])
            self.cluster_whale_weight[cluster] += self.lr_signal * (w_feat["whale"] - self.cluster_whale_weight[cluster])

        self.cluster_ev_weight[cluster] = clamp(self.cluster_ev_weight[cluster], 0.0, 2.0)
        self.cluster_whale_weight[cluster] = clamp(self.cluster_whale_weight[cluster], -1.0, 1.0)

    def preload_history(self, events):
        # NO preload - pure online
        pass