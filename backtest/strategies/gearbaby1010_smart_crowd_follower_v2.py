from collections import defaultdict

from ..core.features import build_features, clamp
from ..core.strategy_base import StrategyBase


class Gearbaby1010SmartCrowdFollowerV2(StrategyBase):
    name = "gearbaby1010_smart_crowd_follower_v2"

    def on_backtest_start(self):
        self.crowd_score = defaultdict(lambda: {"correct": 0.0, "total": 0.0})
        self.crowd_weight = 0.12
        self.surprise_weight = 0.12
        self.base_amount = 70000
        self.ev_mult = 110000
        self.ratio_cut = 0.95
        self.ratio_mult = 0.85
        self.odds_cut = 6.0
        self.odds_mult = 0.7
        self.crowd_bonus = 0.0

    def pick(self, event, state):
        if event.streamer != "gearbaby1010":
            return None

        cs = self.crowd_score[event.streamer]
        acc = (cs["correct"] + 1.0) / (cs["total"] + 2.0)

        best = None
        best_score = -10**9
        for o in event.outcomes:
            f = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
            crowd_term = self.crowd_weight * (f["conf"] - 0.5)
            surprise_term = -self.surprise_weight * max(0.0, f["ratio"] - self.ratio_cut)
            score = (1.0 - acc) * f["ev"] + acc * (crowd_term + surprise_term)
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        if event.streamer != "gearbaby1010":
            return 0.0

        f = build_features(selected, state.prior_win_rate(event.streamer, selected.odds), event_outcomes=event.outcomes)
        amount = self.base_amount + self.ev_mult * f["ev"]
        if f["ratio"] > self.ratio_cut:
            amount *= self.ratio_mult
        if selected.odds > self.odds_cut:
            amount *= self.odds_mult
        if self.crowd_bonus and f["ratio"] <= self.ratio_cut:
            amount += self.crowd_bonus
        return clamp(amount, 20000, 120000)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "gearbaby1010":
            return

        crowd_favorite = max(event.outcomes, key=lambda o: o.points_pct)
        correct = 1.0 if crowd_favorite.outcome_id == event.winner_outcome_id else 0.0

        self.crowd_score[event.streamer]["total"] += 1.0
        self.crowd_score[event.streamer]["correct"] += correct
