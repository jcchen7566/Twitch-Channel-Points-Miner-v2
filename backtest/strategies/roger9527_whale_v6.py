from ..core.strategy_base import StrategyBase


class Roger9527WhaleV6(StrategyBase):
    """
    Always-bet whale pick (backtest twin of runtime roger9527_whale_v6).
    Pick = outcome with MIN ratio (users_pct / points_pct) among odds <= 8.0
    (fallback: min ratio over ALL outcomes, so every event gets a bet).
    Size = 5% of simulated balance (cap 150k) on high-edge window
    (odds 2.5-4.0, ratio 0.75-1.0), else 20k base bet.

    Empirics (roger9527, 711 resolved events, always-bet):
      signal 64k fixed proxy: ROI ~10%, maxDD ~-857k; window bets carry the
      edge (n=34, WR=64.7%, ROI=98.9% isolated). Balance-scaled sizing
      shrinks bets automatically on drawdown, capping real-world risk.
    """

    name = "roger9527_whale_v6"

    MAX_CANDIDATE_ODDS = 8.0
    MIN_ODDS = 2.5
    MAX_ODDS = 4.0
    RATIO_MIN = 0.75
    RATIO_MAX = 1.0
    MAX_SIGNAL_STAKE = 150000
    SIGNAL_FRACTION = 0.05
    BASE_STAKE = 20000
    STARTING_BALANCE = 1280000

    def _ratio(self, o):
        return (o.users_pct / o.points_pct) if o.points_pct > 0 else 99.0

    def on_backtest_start(self):
        self._balance = self.STARTING_BALANCE

    def pick(self, event, state):
        cands = [o for o in event.outcomes if o.odds <= self.MAX_CANDIDATE_ODDS]
        if not cands:
            cands = event.outcomes
        return min(cands, key=self._ratio)

    def size(self, event, selected, state):
        ratio = self._ratio(selected)
        in_window = self.MIN_ODDS <= selected.odds <= self.MAX_ODDS and self.RATIO_MIN <= ratio <= self.RATIO_MAX
        if in_window:
            return min(self._balance * self.SIGNAL_FRACTION, self.MAX_SIGNAL_STAKE, self._balance)
        return min(self.BASE_STAKE, self._balance)

    def on_event_end(self, event, selected, won, pnl, state):
        self._balance += pnl
