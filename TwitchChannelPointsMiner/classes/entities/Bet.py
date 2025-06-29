import copy
import math
from enum import Enum, auto
from random import uniform

from millify import millify

#from TwitchChannelPointsMiner.utils import char_decision_as_index, float_round
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

    def default(self):
        self.strategy = self.strategy if self.strategy is not None else Strategy.SMART
        self.percentage = self.percentage if self.percentage is not None else 5
        self.percentage_gap = (
            self.percentage_gap if self.percentage_gap is not None else 20
        )
        self.max_points = self.max_points if self.max_points is not None else 50000
        self.minimum_points = (
            self.minimum_points if self.minimum_points is not None else 0
        )
        self.stealth_mode = (
            self.stealth_mode if self.stealth_mode is not None else False
        )
        self.delay = self.delay if self.delay is not None else 6
        self.delay_mode = (
            self.delay_mode if self.delay_mode is not None else DelayMode.FROM_END
        )

    def __repr__(self):
        return f"BetSettings(strategy={self.strategy}, percentage={self.percentage}, percentage_gap={self.percentage_gap}, max_points={self.max_points}, minimum_points={self.minimum_points}, stealth_mode={self.stealth_mode})"


class Bet(object):
    __slots__ = ["title", "outcomes", "decision", "total_users", "total_points", "settings", "last_kelly_fraction"]

    def __init__(self, title: str, outcomes: list, settings: BetSettings):
        self.title = title
        self.outcomes = outcomes
        self.__clear_outcomes()
        self.decision: dict = {}
        self.total_users = 0
        self.total_points = 0
        self.settings = settings
        self.last_kelly_fraction = 0.0

    def update_outcomes(self, outcomes):
        for index in range(0, len(self.outcomes)):
            self.outcomes[index][OutcomeKeys.TOTAL_USERS] = int(
                outcomes[index][OutcomeKeys.TOTAL_USERS]
            )
            self.outcomes[index][OutcomeKeys.TOTAL_POINTS] = int(
                outcomes[index][OutcomeKeys.TOTAL_POINTS]
            )
            if outcomes[index]["top_predictors"] != []:
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

        if (
            self.total_users > 0
            and self.total_points > 0
        ):
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
        return f"Bet(total_users={millify(self.total_users)}, total_points={millify(self.total_points)}), decision={self.decision})\n\t\tOutcome A({self.get_outcome(0)})\n\t\tOutcome B({self.get_outcome(1)})"

    def get_decision(self, parsed=False):
        decision = self.outcomes[self.decision["choice"]]
        return decision if parsed is False else Bet.__parse_outcome(decision)

    @staticmethod
    def __parse_outcome(outcome):
        return f"{outcome['title']} ({outcome['color']}), Points: {millify(outcome[OutcomeKeys.TOTAL_POINTS])}, Users: {millify(outcome[OutcomeKeys.TOTAL_USERS])} ({outcome[OutcomeKeys.PERCENTAGE_USERS]}%), Odds: {outcome[OutcomeKeys.ODDS]} ({outcome[OutcomeKeys.ODDS_PERCENTAGE]}%)"

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
        largest=0
        for index in range(0, len(self.outcomes)):
            if self.outcomes[index][key] > self.outcomes[largest][key]:
                largest = index
        return largest

    def __return_number_choice(self, number) -> int:
        if (len(self.outcomes) > number):
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
                compared_value = (
                    self.outcomes[0][fixed_key] + self.outcomes[1][fixed_key]
                )
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

    def __smart_decision_v5_pro(self):
        """
        完美版決策演算法: 結合時間衰減數據、貝葉斯平滑、期望值(EV)、凱利公式與「標題模糊匹配」
        """
        stats_manager = OutcomeStatsManager()
        main_title = self.title

        scores = []
        best_bet_info = {"index": 0, "kelly_f": 0.0, "ev": -999.0}

        for i, outcome in enumerate(self.outcomes):
            title = outcome["title"]
            odds = outcome[OutcomeKeys.ODDS]

            # 1. 透過模糊匹配獲取有效的歷史數據 (不怕標題手打的微小差異)
            stats_obj = stats_manager.get_outcome_stats(main_title, title)
            
            if stats_obj:
                wins_w, total_w = stats_obj.get_effective_stats()
            else:
                wins_w, total_w = 0.0, 0.0

            # 2. 貝葉斯平滑勝率估計 (避免 0 數據的情況)
            # 假設基礎狀態為 45% 勝率 (0.9/2.0)，隨後由主播真實表現修正
            prior_wins = 0.9  
            prior_total = 2.0
            win_rate = (wins_w + prior_wins) / (total_w + prior_total)

            # 3. 計算期望值 Expected Value (EV)
            # EV = (勝率 * 賠率) - 1。EV > 0 代表長期下注會獲利
            ev = (win_rate * odds) - 1

            # 4. 凱利公式計算最佳下注比例
            # f = (bp - q) / b ，其中 b = odds - 1, p = 勝率, q = 敗率
            if odds > 1:
                b = odds - 1
                p = win_rate
                q = 1 - p
                kelly_f = (b * p - q) / b
            else:
                kelly_f = 0.0

            # 評分標準：優先選擇高 EV 選項，若 EV 為負則大幅降分
            score = ev if ev > 0 else ev * 2
            scores.append(score)

            if ev > best_bet_info["ev"] or (ev == best_bet_info["ev"] and score > -999.0):
                best_bet_info = {"index": i, "kelly_f": max(0.0, kelly_f), "ev": ev}

        best_index = scores.index(max(scores))
        
        # 儲存凱利建議金額比例，供 calculate 方法分配點數
        self.last_kelly_fraction = best_bet_info["kelly_f"]
        
        return best_index

    def calculate(self, balance: int) -> dict:
        self.decision = {"choice": None, "amount": 0, "id": None}
        
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
            self.decision["choice"] = self.__smart_decision_v5_pro()
            
        if self.decision["choice"] is not None:
            index = self.decision["choice"]
            self.decision["id"] = self.outcomes[index]["id"]
            
            # --- 動態控倉機制 ---
            base_amount = int(balance * (self.settings.percentage / 100))
            
            if self.settings.strategy == Strategy.SMART:
                # 使用 0.5 倍凱利公式 (Half-Kelly) 降低破產風險
                kelly_amount = int(balance * self.last_kelly_fraction * 0.5)
                
                if self.last_kelly_fraction > 0:
                    # 當前局勢為正期望值：取凱利推薦值與基礎設定的最高者
                    final_amount = max(kelly_amount, base_amount)
                else:
                    # 當前局勢為負期望值：強烈建議不下注或僅下注最少點數防守
                    final_amount = 0 
            else:
                final_amount = base_amount

            # 確保金額不超過系統設定的最高點數限制
            self.decision["amount"] = min(final_amount, self.settings.max_points)
            
            # Stealth Mode (隱形模式)：避免成為被狙擊的大戶
            if (
                self.settings.stealth_mode is True
                and self.decision["amount"] > 0
                and self.decision["amount"] >= self.outcomes[index][OutcomeKeys.TOP_POINTS]
            ):
                reduce_amount = uniform(1, 5)
                self.decision["amount"] = (
                    self.outcomes[index][OutcomeKeys.TOP_POINTS] - reduce_amount
                )
            
            self.decision["amount"] = max(0, int(self.decision["amount"]))

        return self.decision