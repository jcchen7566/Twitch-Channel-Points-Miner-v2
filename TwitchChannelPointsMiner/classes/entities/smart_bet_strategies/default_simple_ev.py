from .base import SmartBetStrategyBase


class DefaultSimpleEv(SmartBetStrategyBase):
    name = "default_simple_ev"

    def choose(self, bet_obj, details):
        # 硬過濾規則（AGENTS.md 鐵律）
        # 1. 高流動性 (>2M) 不碰 - 太高效、負期望值
        if bet_obj.total_points > 2_000_000:
            return None

        valid = []
        for i, d in enumerate(details):
            odds = d["odds"]
            ratio = d["ratio"]
            ev = d["ev"]

            # 2. 低賠率 (≤1.8) 不碰 - 熱門方向負期望值
            if odds <= 1.8:
                continue

            # 3. EV 太低不碰 - 經過 shrinkage=12 後仍 < 2%
            if ev < 0.02:
                continue

            # 4. 眾人瘋狂追捧 (ratio > 1.2) 不碰 - 群體非理性
            if ratio > 1.2:
                continue

            valid.append((i, d))

        if not valid:
            return None

        return max(valid, key=lambda x: x[1]["ev"])[0]

    def size(self, bet_obj, balance, selected_detail):
        if not bet_obj.settings.use_ev_sizing:
            base_amount = bet_obj.settings.base_bet_amount
            amount = int(max(bet_obj.settings.min_bet_amount, min(base_amount, bet_obj.settings.max_points, balance)))
            return amount, 1.0, "legacy_fixed", "tier_legacy"

        ev_adj = selected_detail["ev"]
        base = bet_obj.settings.base_bet_amount
        odds = selected_detail["odds"]

        if ev_adj < 0.0:
            multiplier = 0.25
            reason = "negative_ev"
            tier = "tier_20k"
        else:
            boosted = min(0.50, ev_adj)
            multiplier = 1.00 + boosted
            reason = "positive_ev"
            if multiplier < 1.15:
                tier = "tier_80k"
            elif multiplier < 1.30:
                tier = "tier_96k"
            else:
                tier = "tier_112k"

        amount = int(base * multiplier)
        amount = max(20000, amount)

        # 高賠率懲罰
        if odds > 4.0:
            amount = int(amount * 0.5)
        elif odds > 2.5:
            amount = int(amount * 0.85)

        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        return amount, float(multiplier), reason, tier