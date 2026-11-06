"""A4: 动量策略（强者恒强 / 买龙头）"""
from src.strategies.base_strategy import ArenaStrategyBase


class MomentumStrategy(ArenaStrategyBase):
    """A4: 动量"""
    params = dict(
        detector_name="momentum",
        account_id="A4",
    )
