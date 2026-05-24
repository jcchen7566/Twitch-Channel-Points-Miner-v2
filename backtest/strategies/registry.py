import importlib
import inspect
from pathlib import Path

from ..core.strategy_base import StrategyBase


EXCLUDE = {"__init__.py", "registry.py", "common.py", "baselines.py"}


def get_strategy_factories(group: str = "baseline"):
    if group != "baseline":
        raise ValueError(f"Unknown strategy group: {group}")

    strategies_dir = Path(__file__).resolve().parent
    factories = {}

    for file in sorted(strategies_dir.glob("*.py")):
        if file.name in EXCLUDE:
            continue
        module = importlib.import_module(f"backtest.strategies.{file.stem}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, StrategyBase) or obj is StrategyBase:
                continue
            name = getattr(obj, "name", file.stem)
            factories[name] = obj

    if not factories:
        raise RuntimeError("No strategy plugins discovered in backtest/strategies")
    return factories
