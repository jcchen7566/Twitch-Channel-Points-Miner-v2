from .base import SmartBetStrategyBase


class Roger9527WhaleV6(SmartBetStrategyBase):
    """
    Always-bet whale pick (runtime twin of backtest roger9527_whale_v6).
    choose() NEVER returns None: every event gets a bet.
    Pick = outcome with MIN ratio (users_pct / points_pct) among odds <= 8.0,
    fallback min ratio over all outcomes.
    size() = 150k on high-edge window (odds 2.5-4.0, ratio 0.75-1.0),
    else 20k base bet. Pure rule, no learning, no DB reads.
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

    def choose(self, bet_obj, details):
        best_index = None
        best_ratio = 99.0
        for i, d in enumerate(details):
            odds = float(d.get("odds", 0) or 0)
            if odds <= 1.0 or odds > self.MAX_CANDIDATE_ODDS:
                continue
            ratio = float(d.get("ratio", 0) or 0)
            if ratio < best_ratio:
                best_ratio = ratio
                best_index = i
        if best_index is None:
            # No odds<=8.0 outcome: fall back to min ratio over all outcomes
            best_ratio = 99.0
            for i, d in enumerate(details):
                ratio = float(d.get("ratio", 0) or 0)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_index = i
        return best_index

    def size(self, bet_obj, balance, selected_detail):
        odds = float(selected_detail.get("odds", 0) or 0)
        ratio = float(selected_detail.get("ratio", 0) or 0)
        in_window = self.MIN_ODDS <= odds <= self.MAX_ODDS and self.RATIO_MIN <= ratio <= self.RATIO_MAX
        if in_window:
            # ponytail: balance-scaled, shrinks automatically on drawdown
            amount = min(int(balance * self.SIGNAL_FRACTION), self.MAX_SIGNAL_STAKE, bet_obj.settings.max_points, balance)
        else:
            amount = min(self.BASE_STAKE, balance)
        amount = max(10, amount)
        return int(amount), 1.0, f"whale_v6_{'signal' if in_window else 'base'}_{int(amount)}", "tier_whale_v6"
