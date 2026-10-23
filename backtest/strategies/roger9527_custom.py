import math
from collections import defaultdict
from ..core.features import clamp, liquidity_bucket, odds_regime
from ..core.strategy_base import StrategyBase, OnlineStats


class Roger9527Custom(StrategyBase):
    """
    Custom strategy for roger9527:
    - Title classification (profitable vs unprofitable patterns)
    - Clean online learning (no look-ahead hw/ht features)
    - Ensemble: high-WR signal + EV signal + whale signal
    - Adaptive sizing based on confidence and title quality
    """
    name = "roger9527_custom"

    # Title patterns that are profitable (from analysis)
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
        # Online logistic regression weights (6 features)
        self.w = [0.0] * 6
        self.lr = 0.03
        self.reg = 1e-4

        # Per-title performance tracking
        self.title_stats = defaultdict(lambda: {"wins": 0, "total": 0, "gain": 0.0, "bet": 0.0})

        # Signal weights (learned online)
        self.ev_weight = 0.5
        self.whale_weight = 0.3
        self.contrarian_weight = 0.0
        self.lr_signal = 0.15

    def _title_quality(self, title: str) -> float:
        """Return quality score for title: >0 good, <0 bad, 0 neutral"""
        for kw in self.GOOD_TITLE_KEYWORDS:
            if kw in title:
                return 1.0
        for kw in self.BAD_TITLE_KEYWORDS:
            if kw in title:
                return -1.0
        return 0.0

    def _clean_features(self, o, prior_p, event_outcomes):
        """Build features WITHOUT look-ahead hw/ht"""
        ratio = (o.users_pct / o.points_pct) if o.points_pct > 0 else 1.0
        log_odds = math.log(max(1.01, o.odds))
        implied_p = 1.0 / o.odds
        crowd_p = o.points_pct / 100.0
        surprise = crowd_p - implied_p

        # Use ONLY event-time prior (no hw/ht)
        p = clamp(0.3 * prior_p + 0.7 * implied_p, 0.01, 0.99)
        ev = p * o.odds - 1.0

        # Relative features within event
        rel_ratio = 0.0
        if event_outcomes and len(event_outcomes) > 1:
            all_ratios = [(x.users_pct / x.points_pct) if x.points_pct > 0 else 1.0 for x in event_outcomes]
            mu_r = sum(all_ratios) / len(all_ratios)
            std_r = math.sqrt(sum((x - mu_r)**2 for x in all_ratios) / len(all_ratios)) + 1e-6
            rel_ratio = (ratio - mu_r) / std_r

        # Confidence from sample size
        conf = min(1.0, (o.total_points / 1_000_000) * 0.5 + 0.3)

        # Whale signal
        whale_signal = -ratio if ratio < 0.9 else 0.0  # Negative ratio = whale backing

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
        best = None
        best_score = -1e9

        for o in event.outcomes:
            prior = state.prior_win_rate(event.streamer, o.odds)
            feat = self._clean_features(o, prior, event.outcomes)

            # Model probability
            p_model = max(0.02, min(0.98, self._pred(feat)))

            # Ensemble score
            ev_signal = feat["ev"] * self.ev_weight
            whale_signal = feat["whale"] * self.whale_weight
            contrarian_signal = feat["surprise"] * self.contrarian_weight

            # Title quality adjustment
            title_q = self._title_quality(event.main_title)
            title_boost = title_q * 0.5

            score = p_model * o.odds - 1.0 + 0.1 * (ev_signal + whale_signal + contrarian_signal) + title_boost

            if score > best_score:
                best_score = score
                best = o
                best_feat = feat
                best_p_model = p_model

        return best

    def size(self, event, selected, state):
        # Need to recompute features for selected
        prior = state.prior_win_rate(event.streamer, selected.odds)
        feat = self._clean_features(selected, prior, event.outcomes)
        p_model = max(0.02, min(0.98, self._pred(feat)))
        ev = p_model * selected.odds - 1.0

        # Title quality
        title_q = self._title_quality(event.main_title)

        # Confidence from model + sample
        conf = feat["conf"] * (0.5 + 0.5 * abs(p_model - 0.5) * 2)  # Higher when model is confident

        # Base sizing
        base = 20000 + 60000 * conf

        # EV scaling
        ev_amount = 100000 * max(0, ev)

        # Title quality multiplier
        title_mult = 1.0 + max(0, title_q) * 0.5  # Boost good titles
        if title_q < 0:
            title_mult = 0.5  # Reduce bad titles

        amount = (base + ev_amount) * title_mult

        # High odds penalty
        if selected.odds > 4.0:
            amount *= 0.7
        elif selected.odds > 2.5:
            amount *= 0.85

        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        # Update title stats
        ts = self.title_stats[event.main_title]
        ts["total"] += 1
        ts["bet"] += pnl + (pnl > 0 and pnl or 0)  # approximate
        if won:
            ts["wins"] += 1
            ts["gain"] += pnl
        else:
            ts["gain"] -= abs(pnl)

        # Update online model
        prior = state.prior_win_rate(event.streamer, selected.odds)
        feat = self._clean_features(selected, prior, event.outcomes)
        x = [1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]]
        p = max(0.02, min(0.98, self._pred(feat)))
        y = 1.0 if won else 0.0
        g = p - y
        for i in range(len(self.w)):
            self.w[i] -= self.lr * (g * x[i] + self.reg * self.w[i])

        # Update signal weights
        # Reward signals that would have picked the winner
        winner = next((o for o in event.outcomes if o.outcome_id == event.winner_outcome_id), None)
        if winner:
            w_prior = state.prior_win_rate(event.streamer, winner.odds)
            w_feat = self._clean_features(winner, w_prior, event.outcomes)
            self.ev_weight += self.lr_signal * (w_feat["ev"] - self.ev_weight)
            self.whale_weight += self.lr_signal * (w_feat["whale"] - self.whale_weight)
            self.contrarian_weight += self.lr_signal * (w_feat["surprise"] - self.contrarian_weight)

        # Clamp weights
        self.ev_weight = clamp(self.ev_weight, 0.0, 2.0)
        self.whale_weight = clamp(self.whale_weight, -1.0, 1.0)
        self.contrarian_weight = clamp(self.contrarian_weight, -1.0, 1.0)

    def preload_history(self, events):
        """Train online model on historical events (no look-ahead)"""
        for event in events:
            if not event.label_known:
                continue
            winner = next((o for o in event.outcomes if o.outcome_id == event.winner_outcome_id), None)
            if not winner:
                continue
            prior = 0.5  # No prior during preload
            feat = self._clean_features(winner, prior, event.outcomes)
            x = [1.0, feat["p"], feat["log_odds"], feat["ratio"], feat["conf"], feat["ev"]]
            y = 1.0
            p = max(0.02, min(0.98, self._pred(feat)))
            g = p - y
            for i in range(len(self.w)):
                self.w[i] -= self.lr * (g * x[i] + self.reg * self.w[i])