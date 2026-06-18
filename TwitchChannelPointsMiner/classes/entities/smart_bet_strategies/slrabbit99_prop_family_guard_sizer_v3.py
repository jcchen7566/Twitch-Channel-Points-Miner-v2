from datetime import datetime

from .slrabbit99_prop_family_guard_sizer_v2 import Slrabbit99PropFamilyGuardSizerV2


def _is_sunday(ts: str) -> bool:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").weekday() == 6
    except Exception:
        return False


class Slrabbit99PropFamilyGuardSizerV3(Slrabbit99PropFamilyGuardSizerV2):
    """Inherit v2 scoring exactly. Add Sunday sizing cut (-25%) and
    LightWhale sizing boost (+30% when ratio is 0.9-1.0)."""
    name = "slrabbit99_prop_family_guard_sizer_v3"

    def size(self, bet_obj, balance, selected_detail):
        amount, mult, reason, tier = super().size(bet_obj, balance, selected_detail)

        ts = getattr(bet_obj, "created_at", "") or ""
        ratio = float(selected_detail.get("ratio", 1.0) or 1.0)

        if _is_sunday(ts):
            amount = int(amount * 0.75)
            amount = max(bet_obj.settings.min_bet_amount, amount)

        if 0.9 < ratio < 1.0:
            amount = int(amount * 1.30)
            amount = min(amount, bet_obj.settings.max_points, balance)

        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "slrabbit99_prop_family_guard_sizer_v3", "tier_slrabbit99_family_guard"
