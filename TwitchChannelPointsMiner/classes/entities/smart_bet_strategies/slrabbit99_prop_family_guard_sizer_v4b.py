from datetime import datetime

from .slrabbit99_prop_family_guard_sizer_v2 import Slrabbit99PropFamilyGuardSizerV2


def _is_sunday(ts: str) -> bool:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").weekday() == 6
    except Exception:
        return False


class Slrabbit99PropFamilyGuardSizerV4b(Slrabbit99PropFamilyGuardSizerV2):
    """whale_master: ratio-based primary strategy.

    Organizes scoring around ratio zones. LightWhale (0.9-1.0) gets
    heaviest boost. Heavy crowd ratio>=1.2 gets minimal sizing.
    """
    name = "slrabbit99_prop_family_guard_sizer_v4b"

    def _score(self, title, state, odds, ev, ratio):
        rg = self._odds_regime(odds)
        family = self._family(title)
        pp = float(getattr(self, "_last_pp", 50.0))

        score = ev

        if 0.9 < ratio < 1.0:
            score += 0.18
            if pp >= 60:
                score += 0.10
        elif ratio < 0.7:
            score += 0.14
        elif ratio < 0.9:
            score += 0.06
        elif 1.0 <= ratio < 1.2:
            score -= 0.04
        else:
            score -= 0.10

        if family == "CHALLENGE":
            score += 0.05
            if pp >= 50:
                score += 0.08

        score += 0.14 * state.regime_edge[rg]
        return score, family, rg

    def choose(self, bet_obj, details):
        state = self._load_state(bet_obj)
        title = getattr(bet_obj, "title", "") or ""
        best_idx = 0
        best_score = -10**9
        min_odds = min(float(d.get("odds", 99)) for d in details)
        for i, d in enumerate(details):
            odds = float(d.get("odds", 1.01) or 1.01)
            ev = float(d.get("ev", d.get("ev_raw", 0.0)) or 0.0)
            ratio = float(d.get("ratio", 1.0) or 1.0)
            pp = float(d.get("points_pct", 50.0) or 50.0)
            self._last_pp = pp
            is_fav = abs(odds - min_odds) < 0.001

            # Inject is_fav into score inline
            score = ev
            if 0.9 < ratio < 1.0:
                score += 0.18 + (0.10 if pp >= 60 else 0) + (0.08 if is_fav else 0)
            elif ratio < 0.7:
                score += 0.14
            elif ratio < 0.9:
                score += 0.06
            elif 1.0 <= ratio < 1.2:
                score -= 0.04
            else:
                score -= 0.10

            if is_fav:
                score += 0.06

            family = self._family(title)
            if family == "CHALLENGE":
                score += 0.05 + (0.08 if pp >= 50 else 0)

            score += 0.14 * state.regime_edge[self._odds_regime(odds)]

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
        min_odds = 0.0  # not available here, skip fav check

        amount = 40000.0 + 30000.0 * ev + 15000.0 * state.regime_edge[rg]

        if 0.9 < ratio < 1.0:
            amount *= 1.50
            if pp >= 60:
                amount *= 1.25
        elif ratio < 0.7:
            amount *= 1.30
        elif ratio < 0.9:
            amount *= 1.10
        elif 1.0 <= ratio < 1.2:
            amount *= 0.70
        else:
            amount *= 0.40

        if family == "CHALLENGE" and pp >= 50:
            amount *= 1.20

        ts = getattr(bet_obj, "created_at", "") or ""
        if _is_sunday(ts):
            amount = int(amount * 0.75)
            amount = max(bet_obj.settings.min_bet_amount, amount)

        amount = int(max(bet_obj.settings.min_bet_amount, amount))
        amount = min(amount, bet_obj.settings.max_points, balance)
        amount = max(10, amount)
        multiplier = amount / max(1, bet_obj.settings.base_bet_amount)
        return amount, float(multiplier), "slrabbit99_prop_family_guard_sizer_v4b", "tier_slrabbit99_family_guard"
