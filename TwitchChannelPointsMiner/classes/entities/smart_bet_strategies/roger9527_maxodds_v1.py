from .base import SmartBetStrategyBase


class Roger9527MaxOddsV1(SmartBetStrategyBase):
    """
    Runtime twin of backtest roger9527_maxodds_v1.
    pick = max odds <= 6.0 (fallback 8.0, then all)
    size = 1.5% of balance (cap 150k, floor 20k)
    Pure rule, no learning, no DB reads.
    """

    name = "roger9527_maxodds_v1"

    MAX_ODDS = 6.0
    FALLBACK_ODDS = 8.0
    BALANCE_FRACTION = 0.015
    MAX_STAKE = 150000
    BASE_STAKE = 20000

    def choose(self, bet_obj, details):
        cands = [i for i, d in enumerate(details) 
                 if float(d.get("odds", 0) or 0) <= self.MAX_ODDS]
        if not cands:
            cands = [i for i, d in enumerate(details) 
                     if float(d.get("odds", 0) or 0) <= self.FALLBACK_ODDS]
        if not cands:
            cands = list(range(len(details)))
        return max(cands, key=lambda i: float(details[i].get("odds", 0) or 0))

    def size(self, bet_obj, balance, selected_detail):
        amount = int(balance * self.BALANCE_FRACTION)
        amount = max(amount, self.BASE_STAKE)
        amount = min(amount, self.MAX_STAKE, balance, bet_obj.settings.max_points)
        amount = max(10, amount)
        return amount, 1.0, f"maxodds_v1_{amount}", "tier_maxodds_v1"
