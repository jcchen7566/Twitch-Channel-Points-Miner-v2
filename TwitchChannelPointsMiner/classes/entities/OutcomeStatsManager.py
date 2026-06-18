import json
import math
import os
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path


class OutcomeStats:
    def __init__(self, decay_half_life_hours=12):
        self.wins_weighted = 0.0
        self.total_weighted = 0.0
        self.wins_raw = 0.0
        self.total_raw = 0.0
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
        self.total_raw += 1.0
        if is_win:
            self.wins_weighted += 1.0
            self.wins_raw += 1.0

    def get_effective_stats(self):
        self._apply_decay()
        return self.wins_weighted, self.total_weighted

    def get_raw_stats(self):
        return self.wins_raw, self.total_raw

    def win_rate(self):
        self._apply_decay()
        return self.wins_weighted / self.total_weighted if self.total_weighted > 0 else 0.0

    def to_dict(self):
        self._apply_decay()
        return {
            "wins_w": float(self.wins_weighted),
            "total_w": float(self.total_weighted),
            "wins_raw": float(self.wins_raw),
            "total_raw": float(self.total_raw),
            "last_update": float(self.last_update),
        }

    @staticmethod
    def from_dict(data):
        data = data if isinstance(data, dict) else {}
        stats = OutcomeStats()
        stats.wins_weighted = float(data.get("wins_w", data.get("wins", 0.0)))
        stats.total_weighted = float(data.get("total_w", data.get("total", 0.0)))
        stats.wins_raw = float(data.get("wins_raw", stats.wins_weighted))
        stats.total_raw = float(data.get("total_raw", stats.total_weighted))
        stats.last_update = float(data.get("last_update", 0.0) or 0.0)
        return stats


class OutcomeStatsManager:
    _instance = None

    def __new__(cls, filename=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if filename is None:
                filename = cls._default_db_path()
            cls._instance._init(filename)
        return cls._instance

    def _init(self, filename):
        self.db_path = Path(filename)
        self.legacy_file_path = Path(self._legacy_json_path())
        self.data = defaultdict(lambda: defaultdict(OutcomeStats))
        self.load()

    @staticmethod
    def _default_db_path():
        env_path = os.getenv("BET_LOGS_DB_PATH")
        if env_path:
            return os.path.abspath(os.path.expanduser(env_path))
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "bet_logs.sqlite"))

    @staticmethod
    def _legacy_json_path():
        return os.path.join(os.path.dirname(__file__), "outcome_stats.json")

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _ensure_db_schema(self, conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS outcome_stats (
                main_title TEXT NOT NULL,
                outcome_title TEXT NOT NULL,
                streamer TEXT NOT NULL,
                wins_weighted REAL NOT NULL,
                total_weighted REAL NOT NULL,
                wins_raw REAL NOT NULL,
                total_raw REAL NOT NULL,
                last_update REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (main_title, outcome_title)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcome_stats_streamer ON outcome_stats(streamer)"
        )

    @staticmethod
    def _extract_streamer_name(streamer):
        username = getattr(streamer, "username", None)
        if isinstance(username, str) and username.strip():
            return username.strip().lower()

        text = str(streamer or "").strip()
        if not text:
            return ""

        match = re.search(r"username=([^,\)\s]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()

        if "(" in text:
            text = text.split("(", 1)[0].strip()
        return text.lower()

    def _normalize_text(self, text):
        """標準化字串：轉小寫、去頭尾空白、移除特殊符號 (僅保留英數字與中文)"""
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\u4e00-\u9fa5]", "", text)
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

            if target_norm == existing_norm:
                return existing_title

            ratio = SequenceMatcher(None, target_norm, existing_norm).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = existing_title

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

    @staticmethod
    def _row_to_stats(row):
        stats = OutcomeStats()
        stats.wins_weighted = float(row[0] or 0.0)
        stats.total_weighted = float(row[1] or 0.0)
        stats.wins_raw = float(row[2] or 0.0)
        stats.total_raw = float(row[3] or 0.0)
        stats.last_update = float(row[4] or 0.0)
        return stats

    def _ensure_main_bucket(self, main_title):
        if main_title not in self.data:
            self.data[main_title] = defaultdict(OutcomeStats)
        return self.data[main_title]

    def _hydrate_row(self, main_title, outcome_title, streamer, stats):
        bucket = self._ensure_main_bucket(main_title)
        bucket["_streamer"] = streamer
        bucket[outcome_title] = stats

    def _load_from_sqlite(self):
        loaded = 0
        with self._connect() as conn:
            self._ensure_db_schema(conn)
            rows = conn.execute(
                """
                SELECT main_title, outcome_title, streamer,
                       wins_weighted, total_weighted, wins_raw, total_raw, last_update
                FROM outcome_stats
                """
            ).fetchall()
            for row in rows:
                main_title = row[0]
                outcome_title = row[1]
                streamer = self._extract_streamer_name(row[2])
                stats = self._row_to_stats(row[3:8])
                self._hydrate_row(main_title, outcome_title, streamer, stats)
                loaded += 1
        return loaded

    def _load_from_legacy_json(self):
        if not self.legacy_file_path.exists() or self.legacy_file_path.stat().st_size == 0:
            return 0

        try:
            with open(self.legacy_file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            print(f"Warning: failed to load legacy outcome stats JSON: {exc}")
            return 0

        loaded = 0
        if not isinstance(raw, dict):
            return 0

        for main_title, outcomes in raw.items():
            if not isinstance(outcomes, dict):
                continue

            bucket = self._ensure_main_bucket(main_title)
            streamer = self._extract_streamer_name(outcomes.get("_streamer", ""))
            bucket["_streamer"] = streamer

            for outcome_title, stats_data in outcomes.items():
                if outcome_title == "_streamer":
                    continue

                stats = OutcomeStats.from_dict(stats_data)
                if stats.last_update == 0:
                    stats.last_update = time.time()
                bucket[outcome_title] = stats
                loaded += 1

        return loaded

    def _upsert_stats(self, conn, main_title, outcome_title, streamer, stats):
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            """
            INSERT INTO outcome_stats (
                main_title, outcome_title, streamer,
                wins_weighted, total_weighted, wins_raw, total_raw,
                last_update, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(main_title, outcome_title) DO UPDATE SET
                streamer=excluded.streamer,
                wins_weighted=excluded.wins_weighted,
                total_weighted=excluded.total_weighted,
                wins_raw=excluded.wins_raw,
                total_raw=excluded.total_raw,
                last_update=excluded.last_update,
                updated_at=excluded.updated_at
            """,
            (
                main_title,
                outcome_title,
                streamer,
                float(stats.wins_weighted),
                float(stats.total_weighted),
                float(stats.wins_raw),
                float(stats.total_raw),
                float(stats.last_update),
                now,
                now,
            ),
        )

    def _delete_stats(self, conn, main_title, outcome_title):
        conn.execute(
            "DELETE FROM outcome_stats WHERE main_title = ? AND outcome_title = ?",
            (main_title, outcome_title),
        )

    def record(self, streamer, main_title, outcome_title, is_win):
        c_main = self._find_canonical_main_title(main_title)
        bucket = self._ensure_main_bucket(c_main)

        if "_streamer" not in bucket or not bucket["_streamer"]:
            bucket["_streamer"] = self._extract_streamer_name(streamer)

        c_outcome = self._find_canonical_outcome_title(c_main, outcome_title)
        if c_outcome not in bucket:
            bucket[c_outcome] = OutcomeStats()

        stats = bucket[c_outcome]
        stats.record_result(is_win)

        try:
            with self._connect() as conn:
                self._ensure_db_schema(conn)
                self._upsert_stats(conn, c_main, c_outcome, bucket.get("_streamer", ""), stats)
                self.cleanup(conn=conn)
        except Exception as exc:
            print(f"Error writing outcome stats DB: {exc}")

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
        self.data = defaultdict(lambda: defaultdict(OutcomeStats))
        loaded = self._load_from_sqlite()
        if loaded > 0:
            return

        legacy_loaded = self._load_from_legacy_json()
        if legacy_loaded > 0:
            self.save()

    def save(self):
        try:
            with self._connect() as conn:
                self._ensure_db_schema(conn)
                existing_rows = {
                    (row[0], row[1])
                    for row in conn.execute(
                        "SELECT main_title, outcome_title FROM outcome_stats"
                    )
                }
                current_rows = set()

                for main_title, outcomes in self.data.items():
                    streamer = self._extract_streamer_name(outcomes.get("_streamer", ""))
                    for outcome_title, stats in outcomes.items():
                        if outcome_title == "_streamer":
                            continue
                        current_rows.add((main_title, outcome_title))
                        self._upsert_stats(conn, main_title, outcome_title, streamer, stats)

                for main_title, outcome_title in existing_rows - current_rows:
                    self._delete_stats(conn, main_title, outcome_title)
        except Exception as exc:
            print(f"Error saving outcome stats DB: {exc}")

    def cleanup(self, max_age_days=30, min_total=3.0, conn=None):
        now = time.time()
        max_age = max_age_days * 86400

        keys_to_delete = []

        for main_title, outcomes in list(self.data.items()):
            for outcome_title, stats in list(outcomes.items()):
                if outcome_title == "_streamer":
                    continue

                is_old = (now - stats.last_update) > max_age
                is_small = stats.total_weighted < min_total

                if is_old and is_small:
                    keys_to_delete.append((main_title, outcome_title))

        if not keys_to_delete:
            return

        owns_connection = conn is None
        try:
            if owns_connection:
                conn = self._connect()

            self._ensure_db_schema(conn)
            for main_title, outcome_title in keys_to_delete:
                if main_title in self.data and outcome_title in self.data[main_title]:
                    del self.data[main_title][outcome_title]
                self._delete_stats(conn, main_title, outcome_title)

                if main_title in self.data and len(self.data[main_title]) <= 1:
                    del self.data[main_title]

            if owns_connection:
                conn.commit()
        except Exception as exc:
            print(f"Error cleaning outcome stats DB: {exc}")
        finally:
            if owns_connection and conn is not None:
                conn.close()
