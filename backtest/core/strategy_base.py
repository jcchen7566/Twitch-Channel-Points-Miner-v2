from abc import ABC, abstractmethod
from collections import defaultdict

from .features import odds_regime


class OnlineStats:
    def __init__(self):
        self.global_stats = {"n": 0.0, "w": 0.0}
        self.streamer_stats = defaultdict(lambda: {"n": 0.0, "w": 0.0})
        self.streamer_regime_stats = defaultdict(lambda: {"n": 0.0, "w": 0.0})

    def prior_win_rate(self, streamer: str, odds: float) -> float:
        regime = odds_regime(odds)
        sr = self.streamer_regime_stats[(streamer, regime)]
        ss = self.streamer_stats[streamer]
        gg = self.global_stats
        if sr["n"] >= 8:
            return (sr["w"] + 1.0) / (sr["n"] + 2.0)
        if ss["n"] >= 10:
            return (ss["w"] + 1.5) / (ss["n"] + 3.0)
        if gg["n"] > 0:
            return (gg["w"] + 2.0) / (gg["n"] + 4.0)
        return 0.5

    def update(self, streamer: str, odds: float, won: bool) -> None:
        regime = odds_regime(odds)
        self.global_stats["n"] += 1.0
        self.global_stats["w"] += 1.0 if won else 0.0

        s = self.streamer_stats[streamer]
        s["n"] += 1.0
        s["w"] += 1.0 if won else 0.0

        r = self.streamer_regime_stats[(streamer, regime)]
        r["n"] += 1.0
        r["w"] += 1.0 if won else 0.0


class StrategyBase(ABC):
    name: str

    def on_backtest_start(self) -> None:
        return

    def preload_history(self, events: list) -> None:
        return

    @abstractmethod
    def pick(self, event, state: OnlineStats):
        raise NotImplementedError

    @abstractmethod
    def size(self, event, selected, state: OnlineStats) -> float:
        raise NotImplementedError

    def on_event_end(self, event, selected, won: bool, pnl: float, state: OnlineStats) -> None:
        return
