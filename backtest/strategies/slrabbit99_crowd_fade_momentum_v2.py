from ..core.features import build_features, clamp, odds_regime
from ..core.strategy_base import StrategyBase


class Slrabbit99CrowdFadeMomentumV2(StrategyBase):
    name = "slrabbit99_crowd_fade_momentum_v2"

    def on_backtest_start(self):
        self.regime_edge = {k: 0.0 for k in ("A_1p8", "B_2p2", "C_3p0", "D_5p0", "E_gt5")}

    def pick(self, event, state):
        if event.streamer != "slrabbit99":
            return None

        best = None
        best_score = -10**9
        for o in event.outcomes:
            feat = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
            rg = odds_regime(o.odds)
            crowd_edge = 1.0 - min(2.0, feat["ratio"])
            score = feat["ev"] + 0.16 * crowd_edge + 0.14 * self.regime_edge[rg]
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        if event.streamer != "slrabbit99":
            return 0.0

        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds), event_outcomes=event.outcomes)
        rg = odds_regime(selected.odds)
        amount = 80000 + 50000 * feat["ev"] + 40000 * self.regime_edge[rg]

        if feat["ratio"] > 1.0:
            amount *= 0.7
        if selected.odds > 3.0:
            amount *= 0.6

        return clamp(amount, 20000, 120000)

    def on_event_end(self, event, selected, won, pnl, state):
        if event.streamer != "slrabbit99":
            return

        rg = odds_regime(selected.odds)
        reward = 1.0 if won else -1.0
        self.regime_edge[rg] = 0.9 * self.regime_edge[rg] + 0.1 * reward
