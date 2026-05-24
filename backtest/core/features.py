import math
from .types import OutcomeRow


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wr(w: float, t: float, default: float = 0.5) -> float:
    return (w / t) if t > 0 else default


def odds_regime(odds: float) -> str:
    if odds <= 1.8: return "A_1p8"
    if odds <= 2.2: return "B_2p2"
    if odds <= 3.0: return "C_3p0"
    if odds <= 5.0: return "D_5p0"
    return "E_gt5"


def liquidity_bucket(total_points: float) -> str:
    if total_points < 500_000:
        return "L1_low"
    if total_points < 2_000_000:
        return "L2_mid"
    return "L3_high"


def build_features(o: OutcomeRow, prior_p: float, event_outcomes: list[OutcomeRow] = None) -> dict[str, float]:
    """
    Advanced feature engineering for market efficiency analysis.
    """
    # 1. Ratios and Log Transforms
    ratio = (o.users_pct / o.points_pct) if o.points_pct > 0 else 1.0
    log_odds = math.log(max(1.01, o.odds))
    log_liq = math.log10(max(1.0, o.total_points))
    
    # 2. Market Efficiency Signals
    implied_p = 1.0 / o.odds
    crowd_p = o.points_pct / 100.0
    surprise = crowd_p - implied_p 
    
    # 3. Probability Estimates
    p_hist_w = wr(o.hw, o.ht, implied_p)
    p_hist_r = wr(o.rw, o.rt, implied_p)
    # Blend: use implied_p instead of a flat prior_p from OnlineStats
    p = clamp(0.50 * p_hist_w + 0.20 * p_hist_r + 0.30 * implied_p, 0.01, 0.99)
    ev = p * o.odds - 1.0
    
    # 4. Confidence (Sigmoid)
    conf = 1.0 / (1.0 + math.exp(- (o.ht - 15) / 5.0))
    
    # 5. Event-Relative Context (Normalization)
    rel_ratio = 0.0
    rel_odds = 0.0
    rel_surprise = 0.0
    if event_outcomes and len(event_outcomes) > 1:
        # Relative Ratio (Z-score)
        all_ratios = [(x.users_pct / x.points_pct) if x.points_pct > 0 else 1.0 for x in event_outcomes]
        mu_r = sum(all_ratios) / len(all_ratios)
        std_r = math.sqrt(sum((x - mu_r)**2 for x in all_ratios) / len(all_ratios)) + 1e-6
        rel_ratio = (ratio - mu_r) / std_r
        
        # Relative Odds (Z-score)
        all_log_odds = [math.log(max(1.01, x.odds)) for x in event_outcomes]
        mu_o = sum(all_log_odds) / len(all_log_odds)
        std_o = math.sqrt(sum((x - mu_o)**2 for x in all_log_odds) / len(all_log_odds)) + 1e-6
        rel_odds = (log_odds - mu_o) / std_o

        # Relative Surprise (Z-score)
        all_surprises = [(x.points_pct/100.0 - 1.0/x.odds) for x in event_outcomes]
        mu_s = sum(all_surprises) / len(all_surprises)
        std_s = math.sqrt(sum((x - mu_s)**2 for x in all_surprises) / len(all_surprises)) + 1e-6
        rel_surprise = (surprise - mu_s) / std_s

    # 6. Interaction Terms
    momentum = surprise * p_hist_w
    crowd_liq = ratio * log_liq
    
    return {
        "ratio": ratio,
        "rel_ratio": rel_ratio,
        "log_odds": log_odds,
        "rel_odds": rel_odds,
        "efficiency": surprise,
        "rel_surprise": rel_surprise,
        "log_liq": log_liq,
        "p": p,
        "ev": ev,
        "conf": conf,
        "momentum": momentum,
        "crowd_liq": crowd_liq,
        "users_minus_points": o.users_pct - o.points_pct
    }
