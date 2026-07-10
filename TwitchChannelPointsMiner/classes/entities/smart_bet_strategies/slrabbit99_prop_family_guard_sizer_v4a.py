from datetime import datetime

from .slrabbit99_prop_family_guard_sizer_v2 import Slrabbit99PropFamilyGuardSizerV2


_ROUND = 3600  # rounding for integer division sizing


def _is_sunday(ts: str) -> bool:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").weekday() == 6
    except Exception:
        return False


class Slrabbit99PropFamilyGuardSizerV4a(Slrabbit99PropFamilyGuardSizerV2):
    """pp_alpha: points_pct as primary scoring factor.

    ev modulated by pp/50; whale bonuses; challenge synergy.
    Sizing boosted by pp/ratio signals.
    """
    name = "slrabbit99_prop_family_guard_sizer_v4a"

    def _score(self, title, state, odds, ev, ratio):
        crowd_edge = 1.0 - min(2.0, ratio)
        rg = self._odds_regime(odds)
        pp = float(getattr(self, "_last_pp", 50.0))
        family = self._family(title)

        score = ev * (pp / 50.0)
        score += 0.20 * crowd_edge

        if 0.9 < ratio < 1.0:
            score += 0.08
        if ratio < 0.7:
            score += 0.06

        if family == "CHALLENGE":
            score += 0.06
            if pp >= 50:
                score += 0.10

        score += 0.14 * state.regime_edge[rg]
        if ratio >= 1.0:
            score -= 0.06
        return score, family, rg

    def choose(self, bet_obj, details):
        state = self._load_state(bet_obj)
        title = getattr(bet_obj, "title", "") or ""
        best_idx = 0
        best_score = -10**9
        for i, d in enumerate(details):
            odds = float(d.get("odds", 1.01) or 1.01)
            ev = float(d.get("ev", d.get("ev_raw", 0.0)) or 0.0)
            ratio = float(d.get("ratio", 1.0) or 1.0)
            pp = float(d.get("points_pct", 50.0) or 50.0)
            self._last_pp = pp
            score, _, _ = self._score(title, state, odds, ev, ratio)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        state = self._load_state(bet_obj)
        title = getattr(bet_obj, "title", "") or ""
        odds = float(selected_detail.get("odds", 1.01) or 1.01)
        ev = float(selected_detail.get("ev", selected_detail.get("ev_raw", 0.0)) or 0.0)
        ratio = float(selected_detail.get("ratio", 1.0) or 1.0)
        pp = float(selected_detail.get("points_pct", 50.0) or 50.0)
        family = self._family(title)
        rg = self._odds_regime(odds)

        amount = 40000.0 + 30000.0 * ev + 15000.0 * state.regime_edge[rg]

        if pp >= 70:
            amount *= 1.40
        elif pp >= 60:
            amount *= 1.25
        elif pp >= 50:
            amount *= 1.10

        if 0.9 < ratio < 1.0:
            amount *= 1.35
        elif ratio < 0.7:
            amount *= 1.20

        if family == "CHALLENGE" and pp >= 50:
            amount *= 1.20

        if ratio >= 1.0:
            amount *= 0.80
            if ratio >= 1.2:
                amount *= 0.75

        if family == "ER_PROP" and odds > 3.0:
            amount *= 0.45
        if family == "HITS_PROP" and odds > 2.5:
            amount *= 0.30

        ts = getattr(bet_obj, "created_at", "") or ""
        if _is_sunday(ts):
            amount = int(amount * 0.75)
            amount = max(bet_obj.settings.min_bet_amount, amount)

        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "slrabbit99_prop_family_guard_sizer_v4a", "tier_slrabbit99_family_guard"
