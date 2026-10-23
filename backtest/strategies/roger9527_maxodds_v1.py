from ..core.strategy_base import StrategyBase


class Roger9527MaxOddsV1(StrategyBase):
    """
    roger9527 專用：pick = max odds (cap 6.0)，size = 餘額 1.5%（cap 150k，floor 20k）。
    
    實證（roger9527, 842 場，flat 20k 基準 125.8%）：
      max odds<=6: T1 +77.3%, T2 +48.5%, FULL +125.8%
      1.5% balance fraction: ROI ~200%+, min balance ~1.19M (-7%), maxDD ~25%
    場場下注、不破產、高 ROI。
    """

    name = "roger9527_maxodds_v1"

    MAX_ODDS = 6.0
    FALLBACK_ODDS = 8.0
    BALANCE_FRACTION = 0.015  # 1.5%
    MAX_STAKE = 150000
    BASE_STAKE = 20000
    STARTING_BALANCE = 1280000

    def on_backtest_start(self):
        self._balance = self.STARTING_BALANCE

    def pick(self, event, state):
        cands = [o for o in event.outcomes if o.odds <= self.MAX_ODDS]
        if not cands:
            cands = [o for o in event.outcomes if o.odds <= self.FALLBACK_ODDS]
        if not cands:
            cands = event.outcomes
        return max(cands, key=lambda o: o.odds)

    def size(self, event, selected, state):
        stake = self._balance * self.BALANCE_FRACTION
        stake = max(stake, self.BASE_STAKE)
        stake = min(stake, self.MAX_STAKE, self._balance)
        return stake

    def on_event_end(self, event, selected, won, pnl, state):
        self._balance += pnl
