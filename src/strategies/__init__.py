"""策略注册表：供回测入口按名称取用

ArenaStrategyBase 是 Backtrader 的协调者（Orchestrator），
真正计算分布在 signals / portfolio / execution 各层。
子类只声明自己的检测器名与账户号。
"""
from src.strategies.base_strategy import ArenaStrategyBase
from src.strategies.pullback import PullbackStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.oversold import OversoldStrategy
from src.strategies.momentum import MomentumStrategy

STRATEGY_REGISTRY = {
    "pullback":  PullbackStrategy,
    "breakout":  BreakoutStrategy,
    "oversold":  OversoldStrategy,
    "momentum":  MomentumStrategy,
}


def get_strategy_class(name: str):
    """按检测器名称取策略类"""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"未知策略: {name}，可用: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name]


__all__ = [
    "ArenaStrategyBase",
    "STRATEGY_REGISTRY",
    "get_strategy_class",
    "PullbackStrategy",
    "BreakoutStrategy",
    "OversoldStrategy",
    "MomentumStrategy",
]
