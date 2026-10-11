"""
slrabbit99 專屬自適應策略
特徵分析：
- 95% 場次在高流動性 (>2M)，完全避不開
- 最強區：odds 1.8-5.0，ratio 0.9-1.05
- 鯨魚訊號 (<0.9) 此主播無效甚至負向
- EV>5% 區穩定獲利 (+13%)
- 需要：全程參戰、online 學習 ratio 權重、賠率分層動態調整
"""
import math
from collections import defaultdict
from .base import SmartBetStrategyBase


class Slrabbit99AdaptiveV1(SmartBetStrategyBase):
    name = "slrabbit99_adaptive_v1"

    def __init__(self):
        # Online learning weights for regime-specific signals
        self.regime_weights = {
            'A_<=1.8': {'ratio_w': 0.0, 'ev_w': 1.0, 'conf_w': 0.0},
            'B_1.8_2.2': {'ratio_w': 1.5, 'ev_w': 1.2, 'conf_w': 0.5},
            'C_2.2_3.0': {'ratio_w': 2.0, 'ev_w': 1.5, 'conf_w': 1.0},
            'D_3.0_5.0': {'ratio_w': 2.5, 'ev_w': 2.0, 'conf_w': 1.5},
            'E_>5.0': {'ratio_w': 1.0, 'ev_w': 1.0, 'conf_w': 0.5},
        }

        # Per-title performance (exponential moving average)
        self.title_ema = defaultdict(lambda: {'wr': 0.5, 'n': 0, 'roi': 0.0})

        # Ratio sweet spot tracker (online)
        self.ratio_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'roi_sum': 0.0})

        # Learning rate
        self.lr = 0.1
        self.ema_alpha = 0.15

    def _odds_regime(self, odds: float) -> str:
        if odds <= 1.8:
            return 'A_<=1.8'
        if odds <= 2.2:
            return 'B_1.8_2.2'
        if odds <= 3.0:
            return 'C_2.2_3.0'
        if odds <= 5.0:
            return 'D_3.0_5.0'
        return 'E_>5.0'

    def _ratio_bucket(self, ratio: float) -> str:
        if ratio < 0.9:
            return 'whale'
        if ratio < 1.05:
            return 'sweet'
        if ratio < 1.2:
            return 'neutral'
        if ratio < 1.5:
            return 'crowd'
        return 'extreme'

    def _title_quality(self, title: str) -> float:
        """slrabbit99 標題關鍵字信號較弱，給中性權重"""
        good_kw = ['兩張', '怎麼說', '先死', '前25', '出國', '60%', '鈴珠', '結果', '關盤', '退']
        bad_kw = ['這把會', '瘸瘸', '北投', '底迪', '這輪', '先?', '哪個先', '真真正正', '藍寶堅尼', '夥伴獵']
        for kw in good_kw:
            if kw in title:
                return 0.3
        for kw in bad_kw:
            if kw in title:
                return -0.3
        return 0.0

    def choose(self, bet_obj, details):
        # slrabbit99 全程參戰，不硬過濾
        best_idx = None
        best_score = -1e9

        for i, d in enumerate(details):
            odds = d["odds"]
            ratio = d["ratio"]
            ev = d["ev"]
            conf = d["confidence"]

            regime = self._odds_regime(odds)
            weights = self.regime_weights[regime]
            bucket = self._ratio_bucket(ratio)

            # Ratio signal from online stats
            rs = self.ratio_stats[bucket]
            ratio_signal = 0.0
            if rs['total'] >= 5:
                bucket_wr = rs['wins'] / rs['total']
                bucket_roi = rs['roi_sum'] / rs['total']
                ratio_signal = (bucket_wr - 0.5) * 2 + bucket_roi * 0.5

            # Title EMA signal
            title_sig = self._title_quality(bet_obj.main_title)
            te = self.title_ema[bet_obj.main_title]
            if te['n'] >= 3:
                title_sig = (te['wr'] - 0.5) * 1.5 + te['roi'] * 0.5

            # Composite score
            score = (
                ev * weights['ev_w'] +
                ratio_signal * weights['ratio_w'] +
                (conf - 0.5) * 2 * weights['conf_w'] +
                title_sig
            )

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx if best_idx is not None else 0

    def size(self, bet_obj, balance, selected_detail):
        ev = selected_detail["ev"]
        odds = selected_detail["odds"]
        ratio = selected_detail["ratio"]
        conf = selected_detail["confidence"]
        regime = self._odds_regime(odds)

        # Base sizing by regime
        base_by_regime = {
            'A_<=1.8': 25000,
            'B_1.8_2.2': 35000,
            'C_2.2_3.0': 45000,
            'D_3.0_5.0': 55000,
            'E_>5.0': 30000,
        }
        base = base_by_regime.get(regime, 40000)

        # EV scaling
        if ev < 0:
            multiplier = 0.3
        else:
            multiplier = 1.0 + min(0.6, ev * 3)  # EV 0.05 -> 1.15x, EV 0.2 -> 1.6x

        # Ratio sweet spot bonus
        bucket = self._ratio_bucket(ratio)
        rs = self.ratio_stats[bucket]
        if rs['total'] >= 10 and rs['roi_sum'] / rs['total'] > 0.05:
            multiplier *= 1.2
        elif rs['total'] >= 10 and rs['roi_sum'] / rs['total'] < -0.05:
            multiplier *= 0.7

        # Confidence
        multiplier *= (0.7 + 0.6 * conf)

        amount = int(base * multiplier)
        amount = max(20000, min(amount, 100000))  # cap at 100k
        amount = min(amount, bet_obj.settings.max_points, balance)

        tier_map = {
            'A_<=1.8': 'tier_30k',
            'B_1.8_2.2': 'tier_40k',
            'C_2.2_3.0': 'tier_50k',
            'D_3.0_5.0': 'tier_60k',
            'E_>5.0': 'tier_35k',
        }
        return amount, float(multiplier), f"slrabbit99_{regime}", tier_map.get(regime, 'tier_40k')

    def on_event_end(self, bet_obj, selected_detail, won: bool, pnl: float):
        """Online update - called after each event resolution"""
        odds = selected_detail["odds"]
        ratio = selected_detail["ratio"]
        ev = selected_detail["ev"]
        conf = selected_detail["confidence"]
        regime = self._odds_regime(odds)
        bucket = self._ratio_bucket(ratio)

        # Update ratio stats
        rs = self.ratio_stats[bucket]
        rs['total'] += 1
        if won:
            rs['wins'] += 1
        roi = pnl / selected_detail.get('bet_amount', 1) if selected_detail.get('bet_amount', 1) > 0 else 0
        rs['roi_sum'] += roi

        # Update title EMA
        te = self.title_ema[bet_obj.main_title]
        te['n'] += 1
        alpha = self.ema_alpha
        te['wr'] = (1 - alpha) * te['wr'] + alpha * (1.0 if won else 0.0)
        te['roi'] = (1 - alpha) * te['roi'] + alpha * roi

        # Update regime weights (slow adaptation)
        w = self.regime_weights[regime]
        reward = 1.0 if won else -1.0
        w['ratio_w'] = max(0.0, min(3.0, w['ratio_w'] + self.lr * 0.01 * reward))
        w['ev_w'] = max(0.5, min(3.0, w['ev_w'] + self.lr * 0.01 * reward * (1 if ev > 0 else -1)))
        w['conf_w'] = max(0.0, min(2.0, w['conf_w'] + self.lr * 0.01 * reward * (conf - 0.5)))