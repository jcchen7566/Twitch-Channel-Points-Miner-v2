from ..core.features import build_features, odds_regime
from ..core.strategy_base import StrategyBase


class CrowdFadeMomentum(StrategyBase):
    name = "crowd_fade_momentum"

    def on_backtest_start(self):
        self.regime_edge = {k: 0.0 for k in ("A_1p8", "B_2p2", "C_3p0", "D_5p0", "E_gt5")}

    def pick(self, event, state):
        best = None
        best_score = -10**9
        for o in event.outcomes:
            feat = build_features(o, state.prior_win_rate(event.streamer, o.odds))
            rg = odds_regime(o.odds)
            # crowd fade: prefer low users%/points% when regime recently profitable
            crowd_edge = (1.0 - min(2.0, feat["ratio"]))
            score = feat["ev"] + 0.12 * crowd_edge + 0.18 * self.regime_edge[rg]
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        feat = build_features(selected, state.prior_win_rate(event.streamer, selected.odds))
        rg = odds_regime(selected.odds)
        amount = 70000 + 70000 * feat["ev"] + 30000 * self.regime_edge[rg]
        if feat["ratio"] > 1.2:
            amount *= 0.8
        return amount

    def on_event_end(self, event, selected, won, pnl, state):
        rg = odds_regime(selected.odds)
        reward = 1.0 if won else -1.0
        self.regime_edge[rg] = 0.9 * self.regime_edge[rg] + 0.1 * reward
