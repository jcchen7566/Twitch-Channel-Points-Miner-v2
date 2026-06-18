import math

from ..core.features import clamp
from ..core.strategy_base import StrategyBase


class Bbbb87OnlineModelV2(StrategyBase):
    name = "bbbb87_online_model_v2"

    def on_backtest_start(self):
        self.w = [0.0] * 6
        self.lr = 0.03
        self.reg = 1e-5
        self.low_p = 0.4
        self.base_amount = 80000
        self.ev_mult = 100000
        self.high_odds_pen = 0.6

    @staticmethod
    def _confidence(sample_size: float, odds: float) -> float:
        if sample_size <= 0:
            return 0.5
        size_factor = min(1.0, math.log1p(sample_size) / 3.5)
        odds_factor = 1.0 if odds <= 3.0 else max(0.70, 1.0 - ((odds - 3.0) * 0.04))
        return max(0.35, min(1.0, size_factor * odds_factor))

    def _feat(self, prior: float, odds: float, ratio: float, confidence: float, ev: float):
        return [1.0, prior, math.log(max(1.01, odds)), ratio, confidence, ev]

    def _pred(self, feat):
        z = sum(a * b for a, b in zip(self.w, feat))
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def pick(self, event, state):
        if event.streamer != "bbbb87":
            return None

        best = None
        best_ev = -10**9
        for o in event.outcomes:
            prior = state.prior_win_rate(event.streamer, o.odds)
            ratio = (o.users_pct / o.points_pct) if o.points_pct > 0 else 1.0
            confidence = self._confidence(o.total_users, o.odds)
            ev_proxy = prior * o.odds - 1.0
            feat = self._feat(prior, o.odds, ratio, confidence, ev_proxy)
            p = max(0.02, min(0.98, self._pred(feat)))
            ev = p * o.odds - 1.0
            if ev > best_ev:
                best_ev = ev
                best = o
        return best

    def size(self, event, selected, state):
        if event.streamer != "bbbb87":
            return 0.0

        prior = state.prior_win_rate(event.streamer, selected.odds)
        ratio = (selected.users_pct / selected.points_pct) if selected.points_pct > 0 else 1.0
        confidence = self._confidence(selected.total_users, selected.odds)
        ev_proxy = prior * selected.odds - 1.0
        feat = self._feat(prior, selected.odds, ratio, confidence, ev_proxy)
        p = max(0.02, min(0.98, self._pred(feat)))
        ev = p * selected.odds - 1.0

        amount = self.base_amount + self.ev_mult * ev
        if p < self.low_p:
            amount *= 0.8
        if selected.odds > 5.0:
            amount *= self.high_odds_pen
        return clamp(amount, 20000, 120000)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "bbbb87":
            return

        prior = state.prior_win_rate(event.streamer, selected.odds)
        ratio = (selected.users_pct / selected.points_pct) if selected.points_pct > 0 else 1.0
        confidence = self._confidence(selected.total_users, selected.odds)
        ev_proxy = prior * selected.odds - 1.0
        feat = self._feat(prior, selected.odds, ratio, confidence, ev_proxy)
        p = max(0.02, min(0.98, self._pred(feat)))
        y = 1.0 if won else 0.0
        g = p - y
        for i in range(len(self.w)):
            self.w[i] -= self.lr * (g * feat[i] + self.reg * self.w[i])
