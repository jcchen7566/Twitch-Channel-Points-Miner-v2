import json
import os
import time
import math
import re
from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path


class OutcomeStats:
    def __init__(self, decay_half_life_hours=12):
        self.wins_weighted = 0.0
        self.total_weighted = 0.0
        self.last_update = 0.0
        self.decay_constant = math.log(2) / (decay_half_life_hours * 3600)

    def _apply_decay(self):
        now = time.time()
        if self.last_update > 0:
            duration = now - self.last_update
            if duration > 0:
                factor = math.exp(-self.decay_constant * duration)
                self.wins_weighted *= factor
                self.total_weighted *= factor
        self.last_update = now

    def record_result(self, is_win: bool):
        self._apply_decay()
        self.total_weighted += 1.0
        if is_win:
            self.wins_weighted += 1.0

    def get_effective_stats(self):
        self._apply_decay()
        return self.wins_weighted, self.total_weighted

    def win_rate(self):
        self._apply_decay()
        return self.wins_weighted / self.total_weighted if self.total_weighted > 0 else 0.0

    def to_dict(self):
        self._apply_decay()
        return {
            "wins_w": float(self.wins_weighted),
            "total_w": float(self.total_weighted),
            "last_update": float(self.last_update),
        }

    @staticmethod
    def from_dict(data):
        stats = OutcomeStats()
        stats.wins_weighted = float(data.get("wins_w", data.get("wins", 0.0)))
        stats.total_weighted = float(data.get("total_w", data.get("total", 0.0)))
        stats.last_update = data.get("last_update", 0.0)
        return stats


class OutcomeStatsManager:
    _instance = None

    def __new__(cls, filename=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if filename is None:
                base_path = os.path.dirname(__file__)
                filename = os.path.join(base_path, "outcome_stats.json")
            cls._instance._init(filename)
        return cls._instance

    def _init(self, filename):
        self.file_path = Path(filename)
        self.data = defaultdict(lambda: defaultdict(OutcomeStats))
        self.load()

    def _normalize_text(self, text):
        """標準化字串：轉小寫、去頭尾空白、移除特殊符號 (僅保留英數字與中文)"""
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
        return text

    def _find_canonical_main_title(self, target_title, threshold=0.75):
        """尋找最相似的歷史主標題 (解決標題手打不一致的問題)"""
        if target_title in self.data:
            return target_title
            
        target_norm = self._normalize_text(target_title)
        if not target_norm:
            return target_title
            
        best_match = target_title
        highest_ratio = 0.0
        
        for existing_title in self.data.keys():
            existing_norm = self._normalize_text(existing_title)
            if not existing_norm:
                continue
                
            # 完全匹配 (去除標點後)
            if target_norm == existing_norm:
                return existing_title
                
            # 模糊匹配
            ratio = SequenceMatcher(None, target_norm, existing_norm).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = existing_title
                
        # 相似度大於閾值 (預設 75%)，視為相同標題
        if highest_ratio >= threshold:
            return best_match
            
        return target_title

    def _find_canonical_outcome_title(self, main_title, target_outcome, threshold=0.85):
        """尋找最相似的選項標題 (處理選項輸入的微小差異)"""
        if main_title not in self.data:
            return target_outcome
            
        outcomes_dict = self.data[main_title]
        if target_outcome in outcomes_dict:
            return target_outcome
            
        target_norm = self._normalize_text(target_outcome)
        if not target_norm:
            return target_outcome
            
        best_match = target_outcome
        highest_ratio = 0.0
        
        for existing_outcome in outcomes_dict.keys():
            if existing_outcome == "_streamer":
                continue
            existing_norm = self._normalize_text(existing_outcome)
            if not existing_norm:
                continue
                
            if target_norm == existing_norm:
                return existing_outcome
                
            ratio = SequenceMatcher(None, target_norm, existing_norm).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = existing_outcome
                
        if highest_ratio >= threshold:
            return best_match
            
        return target_outcome

    def record(self, streamer, main_title, outcome_title, is_win):
        # 使用模糊匹配找到正規化的標題
        c_main = self._find_canonical_main_title(main_title)
        
        if c_main not in self.data:
            self.data[c_main] = {}

        if "_streamer" not in self.data[c_main]:
            self.data[c_main]["_streamer"] = str(streamer)

        # 使用模糊匹配找到正規化的選項
        c_outcome = self._find_canonical_outcome_title(c_main, outcome_title)

        if c_outcome not in self.data[c_main]:
            self.data[c_main][c_outcome] = OutcomeStats()

        self.data[c_main][c_outcome].record_result(is_win)
        self.cleanup()
        self.save()

    def get_outcome_stats(self, main_title, outcome_title):
        """提供給 Bet.py 的新介面：透過模糊比對獲取歷史數據物件"""
        c_main = self._find_canonical_main_title(main_title)
        if c_main not in self.data:
            return None
            
        c_outcome = self._find_canonical_outcome_title(c_main, outcome_title)
        
        stats = self.data[c_main].get(c_outcome)
        if stats and not isinstance(stats, str):
            return stats
        return None

    def get_all(self, main_title):
        c_main = self._find_canonical_main_title(main_title)
        return self.data.get(c_main, {})

    def load(self):
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                for main, outcomes in raw.items():
                    self.data[main] = defaultdict(OutcomeStats)
                    for outcome, stats in outcomes.items():
                        if outcome == "_streamer":
                            self.data[main][outcome] = stats
                        else:
                            obj = OutcomeStats.from_dict(stats)

                            if obj.last_update == 0:
                                obj.last_update = time.time()

                            self.data[main][outcome] = obj

        except json.JSONDecodeError:
            print(f"Warning: {self.file_path} is not a valid JSON file.")

    def save(self):
        serializable = {}
        for main, outcomes in self.data.items():
            serializable[main] = {}
            for outcome, stats in outcomes.items():
                if outcome == "_streamer":
                    serializable[main][outcome] = stats
                else:
                    serializable[main][outcome] = stats.to_dict()

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def cleanup(self, max_age_days=30, min_total=3.0):
        now = time.time()
        max_age = max_age_days * 86400

        to_delete_main = []

        for main_title, outcomes in list(self.data.items()):
            keys_to_delete = []

            for outcome, stats in outcomes.items():
                if outcome == "_streamer":
                    continue

                is_old = (now - stats.last_update) > max_age
                is_small = stats.total_weighted < min_total

                if is_old and is_small:
                    keys_to_delete.append(outcome)

            for k in keys_to_delete:
                del self.data[main_title][k]

            if len(self.data[main_title]) <= 1:
                to_delete_main.append(main_title)

        for m in to_delete_main:
            del self.data[m]

        self.save()