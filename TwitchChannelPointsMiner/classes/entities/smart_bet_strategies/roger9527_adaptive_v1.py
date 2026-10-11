"""
roger9527 專屬自適應策略
特徵分析：
- 66% 場次在高流動性 (>2M)，但低流動性 (<500k) 極強 ROI +84%
- >5.0 賠率極強 ROI +38%（長尾策略核心）
- ratio >1.5 長尾極強 ROI +41%（反向指標：眾人追捧反而贏）
- 標題關鍵字極強區分度：好/壞勝率差 50%+
- EV>5% 區 ROI +31%（最可信區間）
- 鯨魚訊號 <0.9 勝率高但 ROI 負（此主播不適用）
- 需要：標題權重為核心、長尾賠率適應、全流動性覆蓋、反向 ratio 學習
"""
import math
from collections import defaultdict
from .base import SmartBetStrategyBase


class Roger9527AdaptiveV1(SmartBetStrategyBase):
    name = "roger9527_adaptive_v1"

    def __init__(self):
        # 標題關鍵字權重（從歷史數據學習）
        self.title_weights = {
            # 強正向 (WR > 60%)
            '前25': 2.0, '夥伴獵': 1.5, '出國': 1.5, '60%': 1.3,
            '兩張': 1.2, '鈴珠': 1.2, '退': 1.0, '先死': 1.0,
            '哪個先': 0.5, '真真正正': 0.3, '底迪': 0.0, '怎麼說': -0.3,
            '結果': -0.3, '藍寶堅尼': -0.8, '北投': -0.8, '關盤': -1.0,
            '這輪': -1.0, '先?': -1.5, '這把會': -1.8, '瘸瘸': -2.0,
        }

        # Online title performance (EMA)
        self.title_ema = defaultdict(lambda: {'wr': 0.5, 'n': 0, 'roi': 0.0, 'total_bet': 0.0})

        # Regime-specific parameters (online learned)
        self.regime_params = {
            'A_<=1.8': {'base_bet': 25000, 'ev_mult': 1.0, 'ratio_pref': 'low'},      # 低賠高勝率，小注穩健
            'B_1.8_2.2': {'base_bet': 30000, 'ev_mult': 1.2, 'ratio_pref': 'mid'},     # 過渡區
            'C_2.2_3.0': {'base_bet': 40000, 'ev_mult': 1.5, 'ratio_pref': 'high'},    # 主力獲利區
            'D_3.0_5.0': {'base_bet': 50000, 'ev_mult': 2.0, 'ratio_pref': 'high'},    # 核心獲利區
            'E_>5.0': {'base_bet': 35000, 'ev_mult': 3.0, 'ratio_pref': 'extreme'},   # 長尾大賠率
        }

        # Liquidity bucket params
        self.liq_params = {
            'low': {'mult': 0.8, 'ev_floor': 0.03},      # <500k - 高報酬但樣本少
            'med': {'mult': 1.0, 'ev_floor': 0.02},      # 500k-2M
            'high': {'mult': 1.1, 'ev_floor': 0.015},    # >2M - 主戰場
        }

        # Ratio preference learning (online)
        self.ratio_pref = defaultdict(lambda: {'low': 0.0, 'mid': 0.0, 'high': 0.0, 'extreme': 0.0})
        self.lr = 0.08
        self.ema_alpha = 0.12

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

    def _liq_bucket(self, total_points: int) -> str:
        if total_points < 500_000:
            return 'low'
        if total_points < 2_000_000:
            return 'med'
        return 'high'

    def _ratio_bucket(self, ratio: float) -> str:
        if ratio < 0.9:
            return 'low'
        if ratio < 1.2:
            return 'mid'
        if ratio < 1.5:
            return 'high'
        return 'extreme'

    def _title_score(self, title: str) -> float:
        """計算標題綜合分數：靜態關鍵字 + 動態 EMA"""
        # 靜態關鍵字
        static_score = 0.0
        matched = []
        for kw, w in self.title_weights.items():
            if kw in title:
                static_score += w
                matched.append(kw)

        # 動態 EMA 修正
        te = self.title_ema[title]
        dynamic_score = 0.0
        if te['n'] >= 3:
            # WR > 55% 給正向，ROI > 5% 給額外加分
            dynamic_score = (te['wr'] - 0.5) * 3 + max(0, te['roi']) * 2

        return static_score + dynamic_score

    def choose(self, bet_obj, details):
        # roger9527 全程參戰，不硬過濾
        best_idx = None
        best_score = -1e9

        title_score = self._title_score(bet_obj.main_title)
        regime = self._odds_regime(details[0]["odds"]) if details else 'C_2.2_3.0'
        liq_bucket = self._liq_bucket(bet_obj.total_points)
        rp = self.ratio_pref[regime]

        for i, d in enumerate(details):
            odds = d["odds"]
            ratio = d["ratio"]
            ev = d["ev"]
            conf = d["confidence"]

            r_bucket = self._ratio_bucket(ratio)

            # Base EV score
            ev_score = ev * self.regime_params[regime]['ev_mult']

            # Ratio preference score (online learned)
            ratio_score = rp.get(r_bucket, 0.0)

            # Confidence
            conf_score = (conf - 0.5) * 2

            # Title score
            title_bonus = title_score * 0.15

            # High odds bonus (roger9527 長尾獲利核心)
            odds_bonus = 0.0
            if odds > 5.0:
                odds_bonus = 0.8
            elif odds > 3.0:
                odds_bonus = 0.4
            elif odds > 2.2:
                odds_bonus = 0.2

            # Low odds penalty (≤1.8 區雖然高勝率但 ROI 負)
            if odds <= 1.8:
                odds_bonus = -0.5

            # Low liquidity bonus
            liq_bonus = 0.0
            if liq_bucket == 'low':
                liq_bonus = 0.6
            elif liq_bucket == 'med':
                liq_bonus = 0.2

            score = ev_score + ratio_score + conf_score + title_bonus + odds_bonus + liq_bonus

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
        liq_bucket = self._liq_bucket(bet_obj.total_points)

        params = self.regime_params[regime]
        lparams = self.liq_params[liq_bucket]
        base = params['base_bet']

        # EV scaling with regime-specific multiplier
        if ev < lparams['ev_floor']:
            multiplier = 0.25
        else:
            multiplier = 1.0 + min(0.8, (ev - lparams['ev_floor']) * params['ev_mult'] * 5)

        # Title quality multiplier
        title_score = self._title_score(bet_obj.main_title)
        if title_score > 1.0:
            multiplier *= 1.3
        elif title_score > 0:
            multiplier *= 1.1
        elif title_score < -1.0:
            multiplier *= 0.5
        elif title_score < 0:
            multiplier *= 0.7

        # Ratio bucket multiplier (online)
        r_bucket = self._ratio_bucket(ratio)
        rp = self.ratio_pref[regime]
        if rp.get(r_bucket, 0) > 0.3:
            multiplier *= 1.2
        elif rp.get(r_bucket, 0) < -0.3:
            multiplier *= 0.7

        # Confidence
        multiplier *= (0.6 + 0.8 * conf)

        # Liquidity multiplier
        multiplier *= lparams['mult']

        amount = int(base * multiplier)
        amount = max(20000, min(amount, 120000))
        amount = min(amount, bet_obj.settings.max_points, balance)

        tier_map = {
            'A_<=1.8': 'tier_25k',
            'B_1.8_2.2': 'tier_35k',
            'C_2.2_3.0': 'tier_50k',
            'D_3.0_5.0': 'tier_60k',
            'E_>5.0': 'tier_40k',
        }
        return amount, float(multiplier), f"roger9527_{regime}_{liq_bucket}", tier_map.get(regime, 'tier_40k')

    def on_event_end(self, bet_obj, selected_detail, won: bool, pnl: float):
        """Online update"""
        odds = selected_detail["odds"]
        ratio = selected_detail["ratio"]
        ev = selected_detail["ev"]
        conf = selected_detail["confidence"]
        regime = self._odds_regime(odds)
        r_bucket = self._ratio_bucket(ratio)
        liq_bucket = self._liq_bucket(bet_obj.total_points)

        bet_amount = selected_detail.get('bet_amount', 1)
        roi = pnl / bet_amount if bet_amount > 0 else 0
        reward = 1.0 if won else -1.0

        # Update title EMA
        te = self.title_ema[bet_obj.main_title]
        te['n'] += 1
        te['total_bet'] += bet_amount
        alpha = self.ema_alpha
        te['wr'] = (1 - alpha) * te['wr'] + alpha * (1.0 if won else 0.0)
        te['roi'] = (1 - alpha) * te['roi'] + alpha * roi

        # Update ratio preference for this regime
        rp = self.ratio_pref[regime]
        rp[r_bucket] = (1 - self.lr) * rp[r_bucket] + self.lr * reward

        # Update regime params (slow)
        rparams = self.regime_params[regime]
        if ev > lparams['ev_floor'] if (lparams := self.liq_params[liq_bucket]) else 0.02:
            rparams['ev_mult'] = max(0.5, min(4.0, rparams['ev_mult'] + self.lr * 0.005 * reward))
        rparams['base_bet'] = max(15000, min(80000, rparams['base_bet'] + self.lr * 100 * reward))

        # Update liquidity params
        lparams = self.liq_params[liq_bucket]
        lparams['mult'] = max(0.5, min(1.5, lparams['mult'] + self.lr * 0.01 * reward))