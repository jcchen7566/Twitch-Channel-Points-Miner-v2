from ..core.features import build_features
from ..core.strategy_base import StrategyBase


class EmpiricalBayesShrinkage(StrategyBase):
    name = "empirical_bayes_shrinkage"

    def pick(self, event, state):
        best = None
        best_score = -10**9
        for o in event.outcomes:
            prior = state.prior_win_rate(event.streamer, o.odds)
            feat = build_features(o, prior)
            n = max(0.0, o.ht)
            lam = n / (n + 12.0)
            p_shrink = lam * feat["p"] + (1.0 - lam) * prior
            score = p_shrink * o.odds - 1.0
            if score > best_score:
                best_score = score
                best = o
        return best

    def size(self, event, selected, state):
        prior = state.prior_win_rate(event.streamer, selected.odds)
        feat = build_features(selected, prior)
        n = max(0.0, selected.ht)
        lam = n / (n + 12.0)
        p_shrink = lam * feat["p"] + (1.0 - lam) * prior
        ev = p_shrink * selected.odds - 1.0
        return 65000 + 85000 * ev
