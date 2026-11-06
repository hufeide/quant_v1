"""A3: 超跌策略（超跌反弹 / 抄底）"""
from src.strategies.base_strategy import ArenaStrategyBase


class OversoldStrategy(ArenaStrategyBase):
    """A3: 超跌"""
    params = dict(
        detector_name="oversold",
        account_id="A3",
    )
