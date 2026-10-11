"""
slrabbit99 專屬自適應策略 (Backtest 版本)
特徵分析：
- 95% 場次在高流動性 (>2M)，完全避不開
- 最強區：odds 1.8-5.0，ratio 0.9-1.05
- 鯨魚訊號 (<0.9) 此主播無效甚至負向
- EV>5% 區穩定獲利 (+13%)
- 需要：全程參戰、online 學習 ratio 權重、賠率分層動態調整
"""
import math
from collections import defaultdict
from ..core.strategy_base import StrategyBase
from ..core.features import clamp, liquidity_bucket, odds_regime


class Slrabbit99AdaptiveV1(StrategyBase):
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
        good_kw = ['兩張', '怎麼說', '先死', '前25', '出國', '60%', '鈴珠', '結果', '關盤', '退']
        bad_kw = ['這把會', '瘸瘸', '北投', '底迪', '這輪', '先?', '哪個先', '真真正正', '藍寶堅尼', '夥伴獵']
        for kw in good_kw:
            if kw in title:
                return 0.3
        for kw in bad_kw:
            if kw in title:
                return -0.3
        return 0.0

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

        for o in event.outcomes:
            prior = state.prior_win_rate(event.streamer, o.odds)
            feat = self._build_features(o, prior, event.outcomes)

            odds = o.odds
            ratio = feat["ratio"]
            ev = feat["ev"]
            conf = feat["conf"]

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
            title_sig = self._title_quality(event.main_title)
            te = self.title_ema[event.main_title]
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
                best_o = o
                best_feat = feat

        # Store feat for use in size()
        self._last_pick_feat = best_feat
        return best_o

    def size(self, event, selected, state):
        # Use stored feat from pick()
        feat = self._last_pick_feat
        if feat is None:
            # Fallback: recompute
            prior = state.prior_win_rate(event.streamer, selected.odds)
            feat = self._build_features(selected, prior, event.outcomes)

        ev = feat["ev"]
        odds = selected.odds
        ratio = feat["ratio"]
        conf = feat["conf"]
        regime = self._odds_regime(odds)

        base_by_regime = {
            'A_<=1.8': 25000,
            'B_1.8_2.2': 35000,
            'C_2.2_3.0': 45000,
            'D_3.0_5.0': 55000,
            'E_>5.0': 30000,
        }
        base = base_by_regime.get(regime, 40000)

        if ev < 0:
            multiplier = 0.3
        else:
            multiplier = 1.0 + min(0.6, ev * 3)

        bucket = self._ratio_bucket(ratio)
        rs = self.ratio_stats[bucket]
        if rs['total'] >= 10 and rs['roi_sum'] / rs['total'] > 0.05:
            multiplier *= 1.2
        elif rs['total'] >= 10 and rs['roi_sum'] / rs['total'] < -0.05:
            multiplier *= 0.7

        multiplier *= (0.7 + 0.6 * conf)

        amount = int(base * multiplier)
        amount = max(20000, min(amount, 100000))
        return amount

    def on_event_end(self, event, selected, won: bool, pnl: float, state):
        odds = selected.odds
        prior = state.prior_win_rate(event.streamer, odds)
        feat = self._build_features(selected, prior, event.outcomes)

        ratio = feat["ratio"]
        ev = feat["ev"]
        conf = feat["conf"]
        regime = self._odds_regime(odds)
        bucket = self._ratio_bucket(ratio)

        # Update ratio stats
        rs = self.ratio_stats[bucket]
        rs['total'] += 1
        if won:
            rs['wins'] += 1
        bet_amount = getattr(selected, 'bet_amount', 1) or 1
        roi = pnl / bet_amount
        rs['roi_sum'] += roi

        # Update title EMA
        te = self.title_ema[event.main_title]
        te['n'] += 1
        alpha = self.ema_alpha
        te['wr'] = (1 - alpha) * te['wr'] + alpha * (1.0 if won else 0.0)
        te['roi'] = (1 - alpha) * te['roi'] + alpha * roi

        # Update regime weights
        w = self.regime_weights[regime]
        reward = 1.0 if won else -1.0
        w['ratio_w'] = max(0.0, min(3.0, w['ratio_w'] + self.lr * 0.01 * reward))
        w['ev_w'] = max(0.5, min(3.0, w['ev_w'] + self.lr * 0.01 * reward * (1 if ev > 0 else -1)))
        w['conf_w'] = max(0.0, min(2.0, w['conf_w'] + self.lr * 0.01 * reward * (conf - 0.5)))