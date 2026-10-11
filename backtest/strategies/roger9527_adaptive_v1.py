"""
roger9527 專屬自適應策略 (Backtest 版本)
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
from ..core.strategy_base import StrategyBase
from ..core.features import clamp, liquidity_bucket, odds_regime


class Roger9527AdaptiveV1(StrategyBase):
    name = "roger9527_adaptive_v1"

    def __init__(self):
        # 標題關鍵字權重（從歷史數據學習）
        self.title_weights = {
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
            'A_<=1.8': {'base_bet': 25000, 'ev_mult': 1.0, 'ratio_pref': 'low'},
            'B_1.8_2.2': {'base_bet': 30000, 'ev_mult': 1.2, 'ratio_pref': 'mid'},
            'C_2.2_3.0': {'base_bet': 40000, 'ev_mult': 1.5, 'ratio_pref': 'high'},
            'D_3.0_5.0': {'base_bet': 50000, 'ev_mult': 2.0, 'ratio_pref': 'high'},
            'E_>5.0': {'base_bet': 35000, 'ev_mult': 3.0, 'ratio_pref': 'extreme'},
        }

        # Liquidity bucket params
        self.liq_params = {
            'low': {'mult': 0.8, 'ev_floor': 0.03},
            'med': {'mult': 1.0, 'ev_floor': 0.02},
            'high': {'mult': 1.1, 'ev_floor': 0.015},
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
        static_score = 0.0
        for kw, w in self.title_weights.items():
            if kw in title:
                static_score += w

        te = self.title_ema[title]
        dynamic_score = 0.0
        if te['n'] >= 3:
            dynamic_score = (te['wr'] - 0.5) * 3 + max(0, te['roi']) * 2

        return static_score + dynamic_score

    def _build_features(self, o, prior_p, event_outcomes):
        ratio = (o.users_pct / o.points_pct) if o.points_pct > 0 else 1.0
        log_odds = math.log(max(1.01, o.odds))
        implied_p = 1.0 / o.odds
        crowd_p = o.points_pct / 100.0
        surprise = crowd_p - implied_p

        p = clamp(0.3 * prior_p + 0.7 * implied_p, 0.01, 0.99)
        ev = p * o.odds - 1.0

        rel_ratio = 0.0
        if event_outcomes and len(event_outcomes) > 1:
            all_ratios = [(x.users_pct / x.points_pct) if x.points_pct > 0 else 1.0 for x in event_outcomes]
            mu_r = sum(all_ratios) / len(all_ratios)
            std_r = math.sqrt(sum((x - mu_r)**2 for x in all_ratios) / len(all_ratios)) + 1e-6
            rel_ratio = (ratio - mu_r) / std_r

        conf = min(1.0, (o.total_points / 1_000_000) * 0.5 + 0.3)
        whale_signal = -ratio if ratio < 0.9 else 0.0

        return {
            "ratio": ratio,
            "rel_ratio": rel_ratio,
            "log_odds": log_odds,
            "p": p,
            "ev": ev,
            "conf": conf,
            "surprise": surprise,
            "whale": whale_signal,
        }

    def pick(self, event, state):
        best_o = None
        best_score = -1e9
        best_feat = None
        best_regime = None
        best_liq = None
        best_r_bucket = None

        title_score = self._title_score(event.main_title)
        # Use first outcome's odds to determine regime for ratio_pref lookup
        first_odds = event.outcomes[0].odds if event.outcomes else 2.5
        regime = self._odds_regime(first_odds)
        total_pts = event.outcomes[0].total_points if event.outcomes else 1_000_000
        liq_bucket = self._liq_bucket(total_pts)
        rp = self.ratio_pref[regime]

        for o in event.outcomes:
            prior = state.prior_win_rate(event.streamer, o.odds)
            feat = self._build_features(o, prior, event.outcomes)

            odds = o.odds
            ratio = feat["ratio"]
            ev = feat["ev"]
            conf = feat["conf"]

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
                best_o = o
                best_feat = feat
                best_regime = regime
                best_liq = liq_bucket
                best_r_bucket = r_bucket

        # Store for use in size() and on_event_end()
        self._last_pick = (best_feat, best_regime, best_liq, best_r_bucket)
        return best_o

    def size(self, event, selected, state):
        # Use stored values from pick()
        stored = getattr(self, '_last_pick', None)
        if stored:
            feat, regime, liq_bucket, r_bucket = stored
        else:
            # Fallback
            prior = state.prior_win_rate(event.streamer, selected.odds)
            feat = self._build_features(selected, prior, event.outcomes)
            regime = self._odds_regime(selected.odds)
            total_pts = event.outcomes[0].total_points if event.outcomes else 1_000_000
            liq_bucket = self._liq_bucket(total_pts)
            r_bucket = self._ratio_bucket(feat["ratio"])

        ev = feat["ev"]
        odds = selected.odds
        ratio = feat["ratio"]
        conf = feat["conf"]

        params = self.regime_params[regime]
        lparams = self.liq_params[liq_bucket]
        base = params['base_bet']

        if ev < lparams['ev_floor']:
            multiplier = 0.25
        else:
            multiplier = 1.0 + min(0.8, (ev - lparams['ev_floor']) * params['ev_mult'] * 5)

        title_score = self._title_score(event.main_title)
        if title_score > 1.0:
            multiplier *= 1.3
        elif title_score > 0:
            multiplier *= 1.1
        elif title_score < -1.0:
            multiplier *= 0.5
        elif title_score < 0:
            multiplier *= 0.7

        rp = self.ratio_pref[regime]
        if rp.get(r_bucket, 0) > 0.3:
            multiplier *= 1.2
        elif rp.get(r_bucket, 0) < -0.3:
            multiplier *= 0.7

        multiplier *= (0.6 + 0.8 * conf)
        multiplier *= lparams['mult']

        amount = int(base * multiplier)
        amount = max(20000, min(amount, 120000))
        return amount

    def on_event_end(self, event, selected, won: bool, pnl: float, state):
        # Use stored values from pick()
        stored = getattr(self, '_last_pick', None)
        if stored:
            feat, regime, liq_bucket, r_bucket = stored
        else:
            # Fallback
            prior = state.prior_win_rate(event.streamer, selected.odds)
            feat = self._build_features(selected, prior, event.outcomes)
            regime = self._odds_regime(selected.odds)
            total_pts = event.outcomes[0].total_points if event.outcomes else 1_000_000
            liq_bucket = self._liq_bucket(total_pts)
            r_bucket = self._ratio_bucket(feat["ratio"])

        odds = selected.odds
        ratio = feat["ratio"]
        ev = feat["ev"]
        conf = feat["conf"]

        bet_amount = getattr(selected, 'bet_amount', 1) or 1
        roi = pnl / bet_amount
        reward = 1.0 if won else -1.0

        # Update title EMA
        te = self.title_ema[event.main_title]
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