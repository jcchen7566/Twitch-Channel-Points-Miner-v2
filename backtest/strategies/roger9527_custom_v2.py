import math
from collections import defaultdict
from ..core.features import clamp, liquidity_bucket, odds_regime
from ..core.strategy_base import StrategyBase, OnlineStats


class Roger9527CustomV2(StrategyBase):
    """
    roger9527 custom v2:
    - Per-title bandit (no preload, pure online)
    - Clean features (no hw/ht)
    - Ensemble: EV + whale + contrarian signals
    - Fast adaptation per title
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
        # Per-title bandit: track avg reward per outcome type
        self.title_bandits = defaultdict(lambda: defaultdict(lambda: {"n": 0, "avg": 0.0}))
        
        # Global signal weights
        self.ev_weight = 0.5
        self.whale_weight = 0.3
        self.contrarian_weight = 0.0
        self.lr_signal = 0.1

        # Model for initial exploration (no preload)
        self.w = [0.0] * 6
        self.lr = 0.05
        self.reg = 1e-4

    def _title_key(self, title: str) -> str:
        """Group similar titles"""
        for kw in self.GOOD_TITLE_KEYWORDS:
            if kw in title:
                return f"GOOD:{kw}"
        for kw in self.BAD_TITLE_KEYWORDS:
            if kw in title:
                return f"BAD:{kw}"
        return "NEUTRAL"

    def _clean_features(self, o, prior_p, event_outcomes):
        ratio = (o.users_pct / o.points_pct) if o.points_pct > 0 else 1.0
        log_odds = math.log(max(1.01, o.odds))
        implied_p = 1.0 / o.odds
        crowd_p = o.points_pct / 100.0
        surprise = crowd_p - implied_p

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
            "surprise": surprise,
            "whale": whale_signal,
        }

    def _pred(self, feat):
        z = sum(a * b for a, b in zip(self.w, [
            1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]
        ]))
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def pick(self, event, state):
        title_key = self._title_key(event.main_title)
        bandit = self.title_bandits[title_key]

        best = None
        best_score = -1e9

        for o in event.outcomes:
            prior = state.prior_win_rate(event.streamer, o.odds)
            feat = self._clean_features(o, prior, event.outcomes)
            p_model = max(0.02, min(0.98, self._pred(feat)))

            # Bandit score for this outcome's odds regime
            arm = self._arm(o.odds)
            bandit_score = bandit[arm]["avg"] if bandit[arm]["n"] > 2 else 0.0

            # Ensemble score
            ev_signal = feat["ev"] * self.ev_weight
            whale_signal = feat["whale"] * self.whale_weight
            contrarian_signal = feat["surprise"] * self.contrarian_weight

            score = p_model * o.odds - 1.0 + 0.1 * (ev_signal + whale_signal + contrarian_signal) + 0.2 * bandit_score

            if score > best_score:
                best_score = score
                best = o
                best_feat = feat
                best_arm = arm

        # Store for size()
        self._last_feat = best_feat
        self._last_arm = best_arm
        return best

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

    def size(self, event, selected, state):
        feat = self._last_feat
        arm = self._last_arm
        if feat is None:
            prior = state.prior_win_rate(event.streamer, selected.odds)
            feat = self._clean_features(selected, prior, event.outcomes)
            arm = self._arm(selected.odds)

        p_model = max(0.02, min(0.98, self._pred(feat)))
        ev = p_model * selected.odds - 1.0

        title_key = self._title_key(event.main_title)
        bandit = self.title_bandits[title_key]
        bandit_conf = min(1.0, bandit[arm]["n"] / 10.0)

        conf = feat["conf"] * (0.5 + 0.5 * bandit_conf)
        base = 20000 + 60000 * conf
        ev_amount = 100000 * max(0, ev)

        # Title quality
        if title_key.startswith("GOOD"):
            title_mult = 1.3
        elif title_key.startswith("BAD"):
            title_mult = 0.5
        else:
            title_mult = 1.0

        amount = (base + ev_amount) * title_mult

        if selected.odds > 4.0:
            amount *= 0.7
        elif selected.odds > 2.5:
            amount *= 0.85

        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        title_key = self._title_key(event.main_title)
        bandit = self.title_bandits[title_key]
        arm = self._arm(selected.odds)

        # Update bandit
        b = bandit[arm]
        b["n"] += 1
        b["avg"] += (pnl - b["avg"]) / b["n"]

        # Update model
        prior = state.prior_win_rate(event.streamer, selected.odds)
        feat = self._clean_features(selected, prior, event.outcomes)
        x = [1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]]
        p = max(0.02, min(0.98, self._pred(feat)))
        y = 1.0 if won else 0.0
        g = p - y
        for i in range(len(self.w)):
            self.w[i] -= self.lr * (g * x[i] + self.reg * self.w[i])

        # Update signal weights
        winner = next((o for o in event.outcomes if o.outcome_id == event.winner_outcome_id), None)
        if winner:
            w_prior = state.prior_win_rate(event.streamer, winner.odds)
            w_feat = self._clean_features(winner, w_prior, event.outcomes)
            self.ev_weight += self.lr_signal * (w_feat["ev"] - self.ev_weight)
            self.whale_weight += self.lr_signal * (w_feat["whale"] - self.whale_weight)
            self.contrarian_weight += self.lr_signal * (w_feat["surprise"] - self.contrarian_weight)

        self.ev_weight = clamp(self.ev_weight, 0.0, 2.0)
        self.whale_weight = clamp(self.whale_weight, -1.0, 1.0)
        self.contrarian_weight = clamp(self.contrarian_weight, -1.0, 1.0)

    def preload_history(self, events):
        # NO preload - pure online learning
        pass