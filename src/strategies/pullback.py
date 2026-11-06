"""A1: 缩量回踩策略（低吸 / 趋势中继）

子类只声明自己的检测器名与账户号，
交易执行、风控、情绪闸门、仓位约束全部继承 ArenaStrategyBase。
"""
from src.strategies.base_strategy import ArenaStrategyBase


class PullbackStrategy(ArenaStrategyBase):
    """A1: 缩量回踩（低吸 / 趋势中继）"""
    params = dict(
        detector_name="pullback",
        account_id="A1",
    )
