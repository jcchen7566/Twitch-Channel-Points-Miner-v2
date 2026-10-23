from ..core.strategy_base import StrategyBase


class Roger9527WhaleV7(StrategyBase):
    """
    Always-bet whale pick v7 (backtest twin of runtime roger9527_whale_v7).

    Pick: among outcomes with 2.0 <= odds <= 8.0, take the one with MIN ratio
    (users_pct / points_pct). If no outcome has odds >= 2.0, fall back to
    min-ratio among odds <= 8.0, then to min-ratio over ALL outcomes —
    every event gets a bet, never None.

    Size: 5% of balance (cap 150k) when the pick sits in the high-edge window
    (odds 2.5-4.0, ratio 0.75-1.0); otherwise 20k base. Balance-scaled
    sizing shrinks automatically on drawdown.

    Empirics (roger9527, 711 resolved events, always-bet):
      FULL ROI 18.7% (WR 43.9%), T1 37.9%, T2 -1.1% (whale edge decaying in
      recent data — market efficiency), min balance -9.4%, maxDD ~20%,
      longest loss streak 11. Trade-off chosen by user: bet EVERY event.
    """

    name = "roger9527_whale_v7"

    PICK_ODDS_MIN = 2.0
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
        cands = [o for o in event.outcomes if self.PICK_ODDS_MIN <= o.odds <= self.MAX_CANDIDATE_ODDS]
        if not cands:
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