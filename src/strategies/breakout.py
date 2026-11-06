"""A2: 突破策略（平台突破 / 追强）"""
from src.strategies.base_strategy import ArenaStrategyBase


class BreakoutStrategy(ArenaStrategyBase):
    """A2: 突破"""
    params = dict(
        detector_name="breakout",
        account_id="A2",
    )
