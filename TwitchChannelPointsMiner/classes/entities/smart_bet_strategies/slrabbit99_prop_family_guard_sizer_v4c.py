from datetime import datetime

from .slrabbit99_prop_family_guard_sizer_v2 import Slrabbit99PropFamilyGuardSizerV2


def _is_sunday(ts: str) -> bool:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").weekday() == 6
    except Exception:
        return False


class Slrabbit99PropFamilyGuardSizerV4c(Slrabbit99PropFamilyGuardSizerV2):
    """two_factor: minimal model using only pp + ratio.

    No title families, no regime edges. Tests whether simplest
    signals (points_pct, ratio) are sufficient.
    """
    name = "slrabbit99_prop_family_guard_sizer_v4c"

    def _score(self, title, state, odds, ev, ratio):
        pp = float(getattr(self, "_last_pp", 50.0))
        pp_factor = (pp - 50.0) / 100.0
        ratio_inv = (1.0 - min(2.0, ratio)) * 0.20
        return ev + pp_factor + ratio_inv + 0.02, "OTHER", "A_1p8"

    def choose(self, bet_obj, details):
        title = getattr(bet_obj, "title", "") or ""
        best_idx = 0
        best_score = -10**9
        for i, d in enumerate(details):
            odds = float(d.get("odds", 1.01) or 1.01)
            ev = float(d.get("ev", d.get("ev_raw", 0.0)) or 0.0)
            ratio = float(d.get("ratio", 1.0) or 1.0)
            pp = float(d.get("points_pct", 50.0) or 50.0)
            self._last_pp = pp
            score, _, _ = self._score(title, None, odds, ev, ratio)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def size(self, bet_obj, balance, selected_detail):
        odds = float(selected_detail.get("odds", 1.01) or 1.01)
        ev = float(selected_detail.get("ev", selected_detail.get("ev_raw", 0.0)) or 0.0)
        ratio = float(selected_detail.get("ratio", 1.0) or 1.0)
        pp = float(selected_detail.get("points_pct", 50.0) or 50.0)

        amount = 40000.0 + 40000.0 * ev

        if pp >= 60:
            amount *= 1.30
        if pp >= 75:
            amount *= 1.30

        if 0.9 < ratio < 1.0:
            amount *= 1.40
        elif ratio < 0.9:
            amount *= 1.15
        elif ratio >= 1.0:
            amount *= 0.65
            if ratio >= 1.5:
                amount *= 0.50

        ts = getattr(bet_obj, "created_at", "") or ""
        if _is_sunday(ts):
            amount = int(amount * 0.75)
            amount = max(bet_obj.settings.min_bet_amount, amount)

        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "slrabbit99_prop_family_guard_sizer_v4c", "tier_slrabbit99_family_guard"
