import math
from collections import defaultdict
from .base import SmartBetStrategyBase


def cluster_title(title: str) -> str:
    title_norm = title.lower()
    if any(k in title_norm for k in ["lol", "英雄聯盟", "勝敗", "輸贏", "潘森", "這把"]):
        return "REGIME_LOL"
    if any(k in title_norm for k in ["tft", "戰棋", "棋", "第幾"]):
        return "REGIME_TFT"
    if any(k in title_norm for k in ["大富翁", "彈珠", "賭", "過關", "幾場"]):
        return "REGIME_LUCK"
    if any(k in title_norm for k in ["聊", "唱歌", "日常"]):
        return "REGIME_CHAT"
    return "REGIME_VARIETY"


class MultiAlphaRegimeExpertV1(SmartBetStrategyBase):
    """
    Deployed version of multi_alpha_regime_expert.
    Uses Bayesian shrinkage and per-regime alpha weights.
    """
    name = "multi_alpha_regime_expert_v1"

    def __init__(self):
        # Key: regime -> expert_weights [Whale_Alpha, Underdog_Alpha]
        # EV is treated as the primary anchor with weight 1.0
        self.regime_alphas = {
            "REGIME_LOL": [0.35, 0.45],
            "REGIME_TFT": [0.20, 0.30],
            "REGIME_LUCK": [0.10, 0.10],
            "REGIME_CHAT": [0.25, 0.20],
            "REGIME_VARIETY": [0.25, 0.25]
        }

    def _get_score(self, bet_obj, d, alphas):
        # Expert 1: Bayesian Shrinkage EV (details already has p_base which is shrunk)
        # In the production Bet.py, d['ev'] is already based on p_hist with prior_strength=5.0
        ev_shrink = d["ev"]
        
        # Expert 2: Whale Alpha (Points concentration)
        # Using d['ratio'] which is percentage_users / (100/odds)
        # ratio < 1.0 means whales are in (points concentrated)
        whale_signal = (1.0 - d["ratio"])
        
        # Expert 3: Underdog Alpha (Rank 2)
        is_rank2 = 1.0 if d["odds_regime"] != "A_<=1.8" and d["odds"] > 2.0 else 0.0
        
        # Combined score = EV + W1 * Whale + W2 * Underdog
        score = ev_shrink + alphas[0] * whale_signal + alphas[1] * is_rank2
        return score

    def choose(self, bet_obj, details):
        regime = cluster_title(bet_obj.title)
        alphas = self.regime_alphas.get(regime, [0.25, 0.25])
        
        best_index = 0
        best_score = -10**9
        
        for i, d in enumerate(details):
            score = self._get_score(bet_obj, d, alphas)
            if score > best_score:
                best_score = score
                best_index = i
        return best_index

    def size(self, bet_obj, balance, selected_detail):
        regime = cluster_title(bet_obj.title)
        alphas = self.regime_alphas.get(regime, [0.25, 0.25])
        
        score = self._get_score(bet_obj, selected_detail, alphas)
        
        # Adaptive sizing: Base 45k + 150k * edge
        conf = selected_detail["confidence"]
        amount = (45000 + 150000 * max(0, score)) * conf
        
        # Risk caps and constraints
        amount = int(amount)
        amount = max(bet_obj.settings.min_bet_amount, amount)
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)

        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), f"multi_alpha_{regime}", "tier_multi_alpha"
