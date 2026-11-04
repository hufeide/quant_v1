"""四个竞技场策略子类

子类只声明自己的检测器名与账户号，
交易执行、风控、情绪闸门、仓位约束全部继承 ArenaStrategyBase。

重要：必须继承 ArenaStrategyBase（而非 bt.Strategy），
否则 next() 的先卖后买逻辑不会执行。
"""
from strategies.arena_base import ArenaStrategyBase
from src.config import get_config


class PullbackStrategy(ArenaStrategyBase):
    """A1: 缩量回踩（低吸 / 趋势中继）"""
    detector_name = "pullback"
    account_id = "A1"


class BreakoutStrategy(ArenaStrategyBase):
    """A2: 平台突破（追强 / 趋势启动），9月后停用新仓"""
    detector_name = "breakout"
    account_id = "A2"


class OversoldStrategy(ArenaStrategyBase):
    """A3: 超跌反弹（均值回归 / 抄底）"""
    detector_name = "oversold"
    account_id = "A3"


class MomentumStrategy(ArenaStrategyBase):
    """A4: 强者恒强（动量 / 买龙头）"""
    detector_name = "momentum"
    account_id = "A4"


# 策略注册表：供回测入口按名称取用
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
