from abc import ABC, abstractmethod


class SmartBetStrategyBase(ABC):
    name = "base"

    @abstractmethod
    def choose(self, bet_obj, details):
        """Return selected outcome index."""
        raise NotImplementedError

    @abstractmethod
    def size(self, bet_obj, balance, selected_detail):
        """Return tuple(amount:int, multiplier:float, reason:str, tier:str)."""
        raise NotImplementedError
