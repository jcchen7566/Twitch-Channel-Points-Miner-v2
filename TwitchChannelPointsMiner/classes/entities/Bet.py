import copy
import math
import os
import sqlite3
from datetime import datetime
from enum import Enum, auto
from random import uniform

from millify import millify

from TwitchChannelPointsMiner.utils import float_round
from TwitchChannelPointsMiner.classes.entities.OutcomeStatsManager import OutcomeStatsManager


class Strategy(Enum):
    MOST_VOTED = auto()
    HIGH_ODDS = auto()
    PERCENTAGE = auto()
    SMART_MONEY = auto()
    SMART = auto()
    NUMBER_1 = auto()
    NUMBER_2 = auto()
    NUMBER_3 = auto()
    NUMBER_4 = auto()
    NUMBER_5 = auto()
    NUMBER_6 = auto()
    NUMBER_7 = auto()
    NUMBER_8 = auto()

    def __str__(self):
        return self.name


class Condition(Enum):
    GT = auto()
    LT = auto()
    GTE = auto()
    LTE = auto()

    def __str__(self):
        return self.name


class OutcomeKeys(object):
    PERCENTAGE_USERS = "percentage_users"
    ODDS_PERCENTAGE = "odds_percentage"
    ODDS = "odds"
    TOP_POINTS = "top_points"
    TOTAL_USERS = "total_users"
    TOTAL_POINTS = "total_points"
    DECISION_USERS = "decision_users"
    DECISION_POINTS = "decision_points"


class DelayMode(Enum):
    FROM_START = auto()
    FROM_END = auto()
    PERCENTAGE = auto()

    def __str__(self):
        return self.name


class FilterCondition(object):
    __slots__ = [
        "by",
        "where",
        "value",
    ]

    def __init__(self, by=None, where=None, value=None, decision=None):
        self.by = by
        self.where = where
        self.value = value

    def __repr__(self):
        return f"FilterCondition(by={self.by.upper()}, where={self.where}, value={self.value})"


class BetSettings(object):
    __slots__ = [
        "strategy",
        "percentage",
        "percentage_gap",
        "max_points",
        "minimum_points",
        "stealth_mode",
        "filter_condition",
        "delay",
        "delay_mode",
        "base_bet_amount",
        "min_bet_amount",
        "max_bet_multiplier",
        "min_bet_multiplier",
        "ev_floor",
        "ev_ceiling",
        "odds_cap_penalty",
        "streamer_modifiers",
        "use_ev_sizing",
    ]

    def __init__(
        self,
        strategy: Strategy = None,
        percentage: int = None,
        percentage_gap: int = None,
        max_points: int = None,
        minimum_points: int = None,
        stealth_mode: bool = None,
        filter_condition: FilterCondition = None,
        delay: float = None,
        delay_mode: DelayMode = None,
        base_bet_amount: int = None,
        min_bet_amount: int = None,
        max_bet_multiplier: float = None,
        min_bet_multiplier: float = None,
        ev_floor: float = None,
        ev_ceiling: float = None,
        odds_cap_penalty: float = None,
        streamer_modifiers: dict = None,
        use_ev_sizing: bool = None,
    ):
        self.strategy = strategy
        self.percentage = percentage
        self.percentage_gap = percentage_gap
        self.max_points = max_points
        self.minimum_points = minimum_points
        self.stealth_mode = stealth_mode
        self.filter_condition = filter_condition
        self.delay = delay
        self.delay_mode = delay_mode
        self.base_bet_amount = base_bet_amount
        self.min_bet_amount = min_bet_amount
        self.max_bet_multiplier = max_bet_multiplier
        self.min_bet_multiplier = min_bet_multiplier
        self.ev_floor = ev_floor
        self.ev_ceiling = ev_ceiling
        self.odds_cap_penalty = odds_cap_penalty
        self.streamer_modifiers = streamer_modifiers
        self.use_ev_sizing = use_ev_sizing

    def default(self):
        self.strategy = self.strategy if self.strategy is not None else Strategy.SMART
        self.percentage = self.percentage if self.percentage is not None else 5
        self.percentage_gap = self.percentage_gap if self.percentage_gap is not None else 20

        # 兼容舊設定：若未提供 base_bet_amount，保留舊邏輯；SMART 策略則預設用 80,000 當基準
        self.base_bet_amount = (
            self.base_bet_amount
            if self.base_bet_amount is not None
            else 80000
        )

        self.min_bet_amount = self.min_bet_amount if self.min_bet_amount is not None else 20000
        self.max_points = self.max_points if self.max_points is not None else 120000
        self.minimum_points = self.minimum_points if self.minimum_points is not None else 0
        self.stealth_mode = self.stealth_mode if self.stealth_mode is not None else False
        self.delay = self.delay if self.delay is not None else 6
        self.delay_mode = self.delay_mode if self.delay_mode is not None else DelayMode.FROM_END

        # EV 倉位控制
        self.max_bet_multiplier = self.max_bet_multiplier if self.max_bet_multiplier is not None else 1.40
        self.min_bet_multiplier = self.min_bet_multiplier if self.min_bet_multiplier is not None else 0.75
        self.ev_floor = self.ev_floor if self.ev_floor is not None else -0.15
        self.ev_ceiling = self.ev_ceiling if self.ev_ceiling is not None else 0.60
        self.odds_cap_penalty = self.odds_cap_penalty if self.odds_cap_penalty is not None else 0.88

        # 個別 streamer 權重（可擴充）
        self.streamer_modifiers = self.streamer_modifiers if self.streamer_modifiers is not None else {
            "slrabbit99": 1.10,
        }

        self.use_ev_sizing = self.use_ev_sizing if self.use_ev_sizing is not None else True

    def __repr__(self):
        return (
            "BetSettings("
            f"strategy={self.strategy}, percentage={self.percentage}, "
            f"percentage_gap={self.percentage_gap}, max_points={self.max_points}, "
            f"minimum_points={self.minimum_points}, stealth_mode={self.stealth_mode}, "
            f"base_bet_amount={self.base_bet_amount}, use_ev_sizing={self.use_ev_sizing}"
            ")"
        )


class Bet(object):
    __slots__ = [
        "title",
        "outcomes",
        "decision",
        "total_users",
        "total_points",
        "settings",
        "last_kelly_fraction",
        "last_ev",
        "last_win_rate",
        "last_odds",
        "last_trap_status",
        "last_ratio",
        "last_bet_multiplier",
        "last_streamer_modifier",
        "last_confidence",
        "last_size_reason",
        "streamer_name",
        "last_p_adj",
        "last_ev_adj",
        "last_odds_regime",
        "last_entry_rule_passed",
        "last_bet_tier",
        "last_model_version",
        "last_p_base",
        "last_ev_raw",
        "last_odds_eff",
        "last_ev_second_best",
        "last_ev_gap",
    ]

    def __init__(self, title: str, outcomes: list, settings: BetSettings, streamer_name: str = ""):
        self.title = title
        self.outcomes = outcomes
        self.__clear_outcomes()
        self.decision: dict = {}
        self.total_users = 0
        self.total_points = 0
        self.settings = settings
        self.last_kelly_fraction = 0.0
        self.last_ev = 0.0
        self.last_win_rate = 0.0
        self.last_odds = 0.0
        self.last_trap_status = "Safe"
        self.last_ratio = 0.0
        self.last_bet_multiplier = 1.0
        self.last_streamer_modifier = 1.0
        self.last_confidence = 1.0
        self.last_size_reason = "default"
        self.streamer_name = (streamer_name or "").lower()
        self.last_p_adj = 0.0
        self.last_ev_adj = 0.0
        self.last_odds_regime = "unknown"
        self.last_entry_rule_passed = False
        self.last_bet_tier = "unknown"
        self.last_model_version = "v3_ev_adj_slr"
        self.last_p_base = 0.0
        self.last_ev_raw = 0.0
        self.last_odds_eff = 0.0
        self.last_ev_second_best = 0.0
        self.last_ev_gap = 0.0

    def update_outcomes(self, outcomes):
        for index in range(0, len(self.outcomes)):
            self.outcomes[index][OutcomeKeys.TOTAL_USERS] = int(
                outcomes[index][OutcomeKeys.TOTAL_USERS]
            )
            self.outcomes[index][OutcomeKeys.TOTAL_POINTS] = int(
                outcomes[index][OutcomeKeys.TOTAL_POINTS]
            )
            if outcomes[index].get("top_predictors", []) != []:
                outcomes[index]["top_predictors"] = sorted(
                    outcomes[index]["top_predictors"],
                    key=lambda x: x["points"],
                    reverse=True,
                )
                top_points = outcomes[index]["top_predictors"][0]["points"]
                self.outcomes[index][OutcomeKeys.TOP_POINTS] = top_points

        self.total_points = 0
        self.total_users = 0
        for index in range(0, len(self.outcomes)):
            self.total_users += self.outcomes[index][OutcomeKeys.TOTAL_USERS]
            self.total_points += self.outcomes[index][OutcomeKeys.TOTAL_POINTS]

        if self.total_users > 0 and self.total_points > 0:
            for index in range(0, len(self.outcomes)):
                self.outcomes[index][OutcomeKeys.PERCENTAGE_USERS] = float_round(
                    (100 * self.outcomes[index][OutcomeKeys.TOTAL_USERS]) / self.total_users
                )
                self.outcomes[index][OutcomeKeys.ODDS] = float_round(
                    0
                    if self.outcomes[index][OutcomeKeys.TOTAL_POINTS] == 0
                    else self.total_points / self.outcomes[index][OutcomeKeys.TOTAL_POINTS]
                )
                self.outcomes[index][OutcomeKeys.ODDS_PERCENTAGE] = float_round(
                    0
                    if self.outcomes[index][OutcomeKeys.ODDS] == 0
                    else 100 / self.outcomes[index][OutcomeKeys.ODDS]
                )

        self.__clear_outcomes()

    def __repr__(self):
        return (
            f"Bet(total_users={millify(self.total_users)}, total_points={millify(self.total_points)}), decision={self.decision})\n"
            f"\t\tOutcome A({self.get_outcome(0)})\n"
            f"\t\tOutcome B({self.get_outcome(1)})"
        )

    def get_decision(self, parsed=False):
        decision = self.outcomes[self.decision["choice"]]
        return decision if parsed is False else Bet.__parse_outcome(decision)

    @staticmethod
    def __parse_outcome(outcome):
        return (
            f"{outcome['title']} ({outcome['color']}), "
            f"Points: {millify(outcome[OutcomeKeys.TOTAL_POINTS])}, "
            f"Users: {millify(outcome[OutcomeKeys.TOTAL_USERS])} "
            f"({outcome[OutcomeKeys.PERCENTAGE_USERS]}%), "
            f"Odds: {outcome[OutcomeKeys.ODDS]} "
            f"({outcome[OutcomeKeys.ODDS_PERCENTAGE]}%)"
        )

    def get_outcome(self, index):
        return Bet.__parse_outcome(self.outcomes[index])

    def __clear_outcomes(self):
        for index in range(0, len(self.outcomes)):
            keys = copy.deepcopy(list(self.outcomes[index].keys()))
            for key in keys:
                if key not in [
                    OutcomeKeys.TOTAL_USERS,
                    OutcomeKeys.TOTAL_POINTS,
                    OutcomeKeys.TOP_POINTS,
                    OutcomeKeys.PERCENTAGE_USERS,
                    OutcomeKeys.ODDS,
                    OutcomeKeys.ODDS_PERCENTAGE,
                    "title",
                    "color",
                    "id",
                ]:
                    del self.outcomes[index][key]
            for key in [
                OutcomeKeys.PERCENTAGE_USERS,
                OutcomeKeys.ODDS,
                OutcomeKeys.ODDS_PERCENTAGE,
                OutcomeKeys.TOP_POINTS,
            ]:
                if key not in self.outcomes[index]:
                    self.outcomes[index][key] = 0

    def __return_choice(self, key) -> int:
        largest = 0
        for index in range(0, len(self.outcomes)):
            if self.outcomes[index][key] > self.outcomes[largest][key]:
                largest = index
        return largest

    def __return_number_choice(self, number) -> int:
        if len(self.outcomes) > number:
            return number
        else:
            return 0

    def skip(self) -> bool:
        if self.settings.filter_condition is not None:
            key = self.settings.filter_condition.by
            condition = self.settings.filter_condition.where
            value = self.settings.filter_condition.value

            fixed_key = (
                key
                if key not in [OutcomeKeys.DECISION_USERS, OutcomeKeys.DECISION_POINTS]
                else key.replace("decision", "total")
            )
            if key in [OutcomeKeys.TOTAL_USERS, OutcomeKeys.TOTAL_POINTS]:
                compared_value = self.outcomes[0][fixed_key] + self.outcomes[1][fixed_key]
            else:
                outcome_index = self.decision["choice"]
                compared_value = self.outcomes[outcome_index][fixed_key]

            if condition == Condition.GT:
                if compared_value > value:
                    return False, compared_value
            elif condition == Condition.LT:
                if compared_value < value:
                    return False, compared_value
            elif condition == Condition.GTE:
                if compared_value >= value:
                    return False, compared_value
            elif condition == Condition.LTE:
                if compared_value <= value:
                    return False, compared_value
            return True, compared_value
        else:
            return False, 0

    def _log_to_csv(self, balance, final_amount, index):
        pass

    @staticmethod
    def _db_path():
        return "bet_logs.sqlite"

    @staticmethod
    def _ensure_db_schema(conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bet_events (
                event_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                streamer TEXT NOT NULL,
                main_title TEXT NOT NULL,
                status TEXT NOT NULL,
                status_reason TEXT,
                balance_before INTEGER NOT NULL,
                placed_amount INTEGER NOT NULL,
                selected_outcome_idx INTEGER,
                selected_outcome_id TEXT,
                winning_outcome_id TEXT,
                result_type TEXT,
                points_won INTEGER,
                points_gained INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bet_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                outcome_idx INTEGER NOT NULL,
                outcome_id TEXT NOT NULL,
                outcome_title TEXT NOT NULL,
                outcome_users INTEGER NOT NULL,
                outcome_points INTEGER NOT NULL,
                outcome_odds REAL NOT NULL,
                outcome_users_percent REAL NOT NULL,
                outcome_points_percent REAL NOT NULL,
                history_wins_weighted REAL NOT NULL,
                history_total_weighted REAL NOT NULL,
                history_wins_raw REAL NOT NULL DEFAULT 0,
                history_total_raw REAL NOT NULL DEFAULT 0,
                total_users INTEGER NOT NULL,
                total_points INTEGER NOT NULL,
                FOREIGN KEY(event_id) REFERENCES bet_events(event_id)
            )
            """
        )
        # Lightweight migration for old DBs created before raw columns were added.
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(bet_outcomes)")}
        if "history_wins_raw" not in existing_columns:
            conn.execute("ALTER TABLE bet_outcomes ADD COLUMN history_wins_raw REAL NOT NULL DEFAULT 0")
        if "history_total_raw" not in existing_columns:
            conn.execute("ALTER TABLE bet_outcomes ADD COLUMN history_total_raw REAL NOT NULL DEFAULT 0")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bet_outcomes_event_id ON bet_outcomes(event_id)"
        )

    @staticmethod
    def _extract_streamer_name(streamer):
        return getattr(streamer, "username", str(streamer))

    def log_bet_decision(self, event_id, streamer, balance_before, decision, status, status_reason=""):
        selected_idx = decision.get("choice")
        selected_outcome_id = ""
        if selected_idx is not None and 0 <= selected_idx < len(self.outcomes):
            selected_outcome_id = self.outcomes[selected_idx].get("id", "")

        stats_manager = OutcomeStatsManager()
        outcome_rows = []
        for idx, outcome in enumerate(self.outcomes):
            stats_obj = stats_manager.get_outcome_stats(self.title, outcome.get("title", ""))
            wins_w, total_w = stats_obj.get_effective_stats() if stats_obj else (0.0, 0.0)
            wins_raw, total_raw = stats_obj.get_raw_stats() if stats_obj else (0.0, 0.0)
            odds = float(outcome.get(OutcomeKeys.ODDS, 0) or 0)
            outcome_rows.append(
                (
                    str(event_id),
                    idx,
                    outcome.get("id", ""),
                    outcome.get("title", ""),
                    int(outcome.get(OutcomeKeys.TOTAL_USERS, 0)),
                    int(outcome.get(OutcomeKeys.TOTAL_POINTS, 0)),
                    float_round(odds),
                    float_round(outcome.get(OutcomeKeys.PERCENTAGE_USERS, 0)),
                    float_round((100.0 / odds) if odds > 0 else 0.0),
                    float_round(wins_w),
                    float_round(total_w),
                    float_round(wins_raw),
                    float_round(total_raw),
                    int(self.total_users),
                    int(self.total_points),
                )
            )

        try:
            with sqlite3.connect(self._db_path()) as conn:
                self._ensure_db_schema(conn)
                conn.execute(
                    """
                    INSERT INTO bet_events (
                        event_id, created_at, resolved_at, streamer, main_title, status, status_reason,
                        balance_before, placed_amount, selected_outcome_idx, selected_outcome_id,
                        winning_outcome_id, result_type, points_won, points_gained
                    ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
                    ON CONFLICT(event_id) DO UPDATE SET
                        status=excluded.status,
                        status_reason=excluded.status_reason,
                        balance_before=excluded.balance_before,
                        placed_amount=excluded.placed_amount,
                        selected_outcome_idx=excluded.selected_outcome_idx,
                        selected_outcome_id=excluded.selected_outcome_id
                    """,
                    (
                        str(event_id),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self._extract_streamer_name(streamer),
                        self.title,
                        status,
                        status_reason,
                        int(balance_before),
                        int(decision.get("amount", 0)),
                        selected_idx,
                        selected_outcome_id,
                    ),
                )
                conn.execute("DELETE FROM bet_outcomes WHERE event_id = ?", (str(event_id),))
                conn.executemany(
                    """
                    INSERT INTO bet_outcomes (
                        event_id, outcome_idx, outcome_id, outcome_title,
                        outcome_users, outcome_points, outcome_odds,
                        outcome_users_percent, outcome_points_percent,
                        history_wins_weighted, history_total_weighted,
                        history_wins_raw, history_total_raw,
                        total_users, total_points
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    outcome_rows,
                )
        except Exception as e:
            print(f"Error writing decision DB: {e}")

    def log_bet_result(self, event_id, result_type, points_won, points_gained, winning_outcome_id=None):
        try:
            with sqlite3.connect(self._db_path()) as conn:
                self._ensure_db_schema(conn)
                winner_id = str(winning_outcome_id or "")
                conn.execute(
                    """
                    UPDATE bet_events
                    SET resolved_at = ?,
                        result_type = ?,
                        points_won = ?,
                        points_gained = ?,
                        winning_outcome_id = CASE
                            WHEN ? <> '' THEN ?
                            ELSE COALESCE(winning_outcome_id, '')
                        END,
                        status = CASE WHEN status = 'PLACED' THEN ? ELSE status END
                    WHERE event_id = ?
                    """,
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        str(result_type),
                        int(points_won),
                        int(points_gained),
                        winner_id,
                        winner_id,
                        f"RESOLVED_{result_type}",
                        str(event_id),
                    ),
                )
        except Exception as e:
            print(f"Error updating decision DB: {e}")

    def log_winning_outcome_id(self, event_id, winning_outcome_id):
        winner_id = str(winning_outcome_id or "")
        if not winner_id:
            return
        try:
            with sqlite3.connect(self._db_path()) as conn:
                self._ensure_db_schema(conn)
                conn.execute(
                    """
                    UPDATE bet_events
                    SET winning_outcome_id = ?
                    WHERE event_id = ?
                      AND COALESCE(winning_outcome_id, '') = ''
                    """,
                    (winner_id, str(event_id)),
                )
        except Exception as e:
            print(f"Error backfilling winning_outcome_id: {e}")

    def _get_streamer_modifier(self) -> float:
        stats_manager = OutcomeStatsManager()
        streamer_info = "Unknown"
        try:
            all_data = stats_manager.get_all(self.title)
            if all_data and "_streamer" in all_data:
                streamer_info = str(all_data["_streamer"]).lower()
        except Exception:
            pass

        modifier_map = self.settings.streamer_modifiers or {}
        return float(modifier_map.get(streamer_info, 1.0))

    def _compute_confidence(self, sample_size: float, odds: float) -> float:
        if sample_size <= 0:
            return 0.5

        # 小樣本保守，大樣本趨近 1
        size_factor = min(1.0, math.log1p(sample_size) / 3.5)
        odds_factor = 1.0 if odds <= 3.0 else max(0.70, 1.0 - ((odds - 3.0) * 0.04))
        return max(0.35, min(1.0, size_factor * odds_factor))

    def _compute_multiplier_from_ev(self, ev: float) -> tuple[float, str]:
        floor_ev = self.settings.ev_floor
        ceiling_ev = self.settings.ev_ceiling
        min_mult = self.settings.min_bet_multiplier
        max_mult = self.settings.max_bet_multiplier

        if ceiling_ev <= floor_ev:
            ceiling_ev = floor_ev + 0.01

        clipped_ev = max(floor_ev, min(ceiling_ev, ev))
        t = (clipped_ev - floor_ev) / (ceiling_ev - floor_ev)
        multiplier = min_mult + (max_mult - min_mult) * t

        if ev <= floor_ev:
            reason = "low_ev"
        elif ev >= ceiling_ev:
            reason = "high_ev"
        elif ev >= 0.15:
            reason = "strong_ev"
        elif ev >= 0.0:
            reason = "slight_positive_ev"
        else:
            reason = "negative_ev"

        return multiplier, reason

    def _estimate_side_win_rate(
        self,
        raw_odds: float,
        wins_w: float = 0.0,
        total_w: float = 0.0,
        streamer_modifier: float = 1.0,
        trap_bonus: float = 0.0,
    ) -> float:
        implied_prob = (1.0 / raw_odds) if raw_odds > 0 else 0.5

        # 拉回到保守先驗：歷史資料少時，多靠 implied_prob
        prior_strength = 6.0
        posterior = (wins_w + implied_prob * prior_strength) / (total_w + prior_strength)

        posterior *= streamer_modifier
        posterior += trap_bonus

        # 控制在可用範圍，避免把 Kelly / EV 放大到失真
        return max(0.02, min(0.80, posterior))

    @staticmethod
    def _odds_regime(odds: float) -> str:
        if odds <= 1.8:
            return "A_<=1.8"
        if odds <= 2.2:
            return "B_1.8_2.2"
        if odds <= 3.0:
            return "C_2.2_3.0"
        if odds <= 5.0:
            return "D_3.0_5.0"
        return "E_>5.0"

    @staticmethod
    def _crowd_bonus(ratio: float) -> float:
        if ratio < 0.90:
            return 0.04
        if ratio < 1.05:
            return 0.01
        if ratio < 1.20:
            return -0.06
        if ratio < 1.50:
            return -0.02
        return -0.04

    @staticmethod
    def _odds_bonus(odds: float) -> float:
        if odds <= 1.8:
            return 0.02
        if odds <= 2.2:
            return 0.0
        if odds <= 3.0:
            return -0.01
        if odds <= 5.0:
            return -0.03
        return -0.05

    @staticmethod
    def _confidence_bonus(confidence: float) -> float:
        if confidence >= 0.75:
            return 0.02
        if confidence >= 0.60:
            return 0.01
        if confidence < 0.50:
            return -0.02
        return 0.0

    def _slrabbit_entry_pass(self, p_adj: float, ev_adj: float, confidence: float, ratio: float, odds: float) -> bool:
        if odds <= 2.2:
            return ev_adj > 0.0
        if odds <= 3.0:
            return ev_adj >= 0.03 and confidence >= 0.55
        if odds <= 5.0:
            return ev_adj >= 0.08 and confidence >= 0.62 and ratio < 1.05
        return ev_adj >= 0.15 and confidence >= 0.72 and ratio < 0.95

    def _smart_choice(self) -> int:
        stats_manager = OutcomeStatsManager()
        details = []

        for outcome in self.outcomes:
            raw_odds = float(outcome[OutcomeKeys.ODDS] or 0.0)
            if raw_odds <= 1.0:
                raw_odds = 1.01

            current_ratio = (
                outcome.get(OutcomeKeys.PERCENTAGE_USERS, 0)
                / ((100.0 / raw_odds) if raw_odds > 0 else 1.0)
            )

            stats_obj = stats_manager.get_outcome_stats(self.title, outcome["title"])
            wins_w, total_w = stats_obj.get_effective_stats() if stats_obj else (0.0, 0.0)

            implied_prob = 1.0 / raw_odds
            # 最簡單貝葉斯平滑：避免樣本太少時勝率失真
            prior_strength = 5.0
            p_hist = (wins_w + (implied_prob * prior_strength)) / (total_w + prior_strength)
            p_hist = max(0.03, min(0.95, p_hist))

            ev = (p_hist * raw_odds) - 1.0
            confidence = self._compute_confidence(total_w, raw_odds)
            kelly_f = max(0.0, (raw_odds * p_hist - 1) / (raw_odds - 1)) if raw_odds > 1 else 0.0
            details.append(
                {
                    "kelly": kelly_f,
                    "ev": ev,
                    "wr": p_hist,
                    "odds": raw_odds,
                    "trap": "Simple_EV",
                    "ratio": current_ratio,
                    "confidence": confidence,
                    "entry_ok": True,
                    "odds_regime": self._odds_regime(raw_odds),
                    "p_base": p_hist,
                    "ev_raw": ev,
                    "odds_eff": raw_odds,
                }
            )

        best_index = max(range(len(details)), key=lambda idx: details[idx]["ev"])
        sorted_evs = sorted([d["ev"] for d in details], reverse=True)
        second_best = sorted_evs[1] if len(sorted_evs) > 1 else sorted_evs[0]

        d = details[best_index]

        self.last_kelly_fraction = d["kelly"]
        self.last_ev = d["ev"]
        self.last_win_rate = d["wr"]
        self.last_odds = d["odds"]
        self.last_trap_status = d["trap"]
        self.last_ratio = d["ratio"]
        self.last_confidence = d["confidence"]
        self.last_p_adj = d["wr"]
        self.last_ev_adj = d["ev"]
        self.last_odds_regime = d["odds_regime"]
        self.last_entry_rule_passed = d["entry_ok"]
        self.last_model_version = "v4_simple_ev"
        self.last_p_base = d["p_base"]
        self.last_ev_raw = d["ev_raw"]
        self.last_odds_eff = d["odds_eff"]
        self.last_ev_second_best = second_best
        self.last_ev_gap = d["ev"] - second_best

        return best_index

    def _calc_sized_amount(self, balance: int) -> tuple[int, float, str]:
        if not self.settings.use_ev_sizing:
            base_amount = self.settings.base_bet_amount
            return int(max(self.settings.min_bet_amount, min(base_amount, self.settings.max_points, balance))), 1.0, "legacy_fixed"

        ev_adj = self.last_ev_adj
        base = self.settings.base_bet_amount

        # 最簡單版本：負EV最小注；正EV以80k為中心往上放大
        if ev_adj < 0.0:
            multiplier = 0.25
            reason = "negative_ev"
            tier = "tier_20k"
        else:
            boosted = min(0.50, ev_adj)
            multiplier = 1.00 + boosted
            reason = "positive_ev"
            if multiplier < 1.15:
                tier = "tier_80k"
            elif multiplier < 1.30:
                tier = "tier_96k"
            else:
                tier = "tier_112k"

        amount = int(base * multiplier)
        amount = max(20000, amount)
        amount = min(amount, self.settings.max_points, balance)
        self.last_bet_tier = tier

        # 絕對不要超過餘額與系統上限
        amount = max(10, amount)

        return int(amount), float(multiplier), reason

    def calculate(self, balance: int) -> dict:
        self.decision = {"choice": None, "amount": 0, "id": None}

        # 帳戶餘額連最低下注都不夠時，直接放棄
        if balance < 10:
            return self.decision

        # 策略選擇
        if self.settings.strategy == Strategy.MOST_VOTED:
            self.decision["choice"] = self.__return_choice(OutcomeKeys.TOTAL_USERS)
        elif self.settings.strategy == Strategy.HIGH_ODDS:
            self.decision["choice"] = self.__return_choice(OutcomeKeys.ODDS)
        elif self.settings.strategy == Strategy.PERCENTAGE:
            self.decision["choice"] = self.__return_choice(OutcomeKeys.ODDS_PERCENTAGE)
        elif self.settings.strategy == Strategy.SMART_MONEY:
            self.decision["choice"] = self.__return_choice(OutcomeKeys.TOP_POINTS)
        elif self.settings.strategy == Strategy.NUMBER_1:
            self.decision["choice"] = self.__return_number_choice(0)
        elif self.settings.strategy == Strategy.NUMBER_2:
            self.decision["choice"] = self.__return_number_choice(1)
        elif self.settings.strategy == Strategy.NUMBER_3:
            self.decision["choice"] = self.__return_number_choice(2)
        elif self.settings.strategy == Strategy.NUMBER_4:
            self.decision["choice"] = self.__return_number_choice(3)
        elif self.settings.strategy == Strategy.NUMBER_5:
            self.decision["choice"] = self.__return_number_choice(4)
        elif self.settings.strategy == Strategy.NUMBER_6:
            self.decision["choice"] = self.__return_number_choice(5)
        elif self.settings.strategy == Strategy.NUMBER_7:
            self.decision["choice"] = self.__return_number_choice(6)
        elif self.settings.strategy == Strategy.NUMBER_8:
            self.decision["choice"] = self.__return_number_choice(7)
        elif self.settings.strategy == Strategy.SMART:
            self.decision["choice"] = self._smart_choice()

        if self.decision["choice"] is not None:
            index = self.decision["choice"]
            self.decision["id"] = self.outcomes[index]["id"]

            if self.settings.strategy == Strategy.SMART:
                final_amount, multiplier, reason = self._calc_sized_amount(balance)
                self.last_bet_multiplier = multiplier
                self.last_size_reason = reason
            else:
                # 非 SMART 模式維持舊行為
                base_amount = int(balance * (self.settings.percentage / 100))
                final_amount = base_amount
                self.last_bet_multiplier = 1.0
                self.last_size_reason = "legacy_non_smart"

            # 一般保護：不超過 max_points 與餘額
            self.decision["amount"] = min(int(final_amount), self.settings.max_points, balance)

            # 確保至少 10 點
            self.decision["amount"] = max(10, self.decision["amount"])

            # 隱形模式：避免超過大戶
            if (
                self.settings.stealth_mode is True
                and self.decision["amount"] > 0
                and self.decision["amount"] >= self.outcomes[index][OutcomeKeys.TOP_POINTS]
            ):
                reduce_amount = uniform(1, 5)
                self.decision["amount"] = self.outcomes[index][OutcomeKeys.TOP_POINTS] - reduce_amount

            # 浮點修正
            self.decision["amount"] = max(self.settings.min_bet_amount, int(self.decision["amount"]))

        return self.decision
