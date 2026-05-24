import importlib
import inspect
from pathlib import Path

from .base import SmartBetStrategyBase


_EXCLUDE = {"__init__.py", "base.py", "registry.py"}


import sys

def load_strategies() -> dict:
    out = {}
    strategies_dir = Path(__file__).resolve().parent
    for file in sorted(strategies_dir.glob("*.py")):
        if file.name in _EXCLUDE:
            continue
        
        module_name = f"TwitchChannelPointsMiner.classes.entities.smart_bet_strategies.{file.stem}"
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
            
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, SmartBetStrategyBase) or obj is SmartBetStrategyBase:
                continue
            inst = obj()
            out[inst.name] = inst
    return out
