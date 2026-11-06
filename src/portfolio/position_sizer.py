"""仓位计算器：资金、ATR、止损价 -> 目标股数（不直接下单）

职责边界：
    只回答"每只股票买多少股"。不决定买哪些股票（portfolio_manager），
    不下单（order_manager），不修改持仓状态。

核心思想（风险预算法）：
    每笔交易的风险金额 = 总资金 × risk_pct
    止损距离 = ATR × atr_multiple
    目标股数 = 风险金额 / 止损距离，向下取整到整手（100 股）
    再用单股最大仓位（max_position_pct × 资金 / 价格）封顶。

这与 Backtrader 的 bt.Sizer 不同：bt.Sizer 只解决单次买卖的数量，
本模块要同时兼顾组合层的资金分配上限，因此推荐放在 portfolio 层。
"""
from __future__ import annotations

from typing import Dict, Optional


class PositionSizer:
    """ATR 风险预算仓位计算器"""

    def __init__(self, config: Dict):
        config = config or {}
        # 每笔交易风险预算占比（默认 0.5% = 千五）
        self.risk_pct: float = float(config.get("risk_pct", 0.005))
        # 止损距离 = ATR × atr_multiple（默认 2 倍 ATR）
        self.atr_multiple: float = float(config.get("atr_multiple", 2.0))
        # 单股最大仓位占比（封顶）
        self.max_position_pct: float = float(config.get("max_position_pct", 0.18))
        # 整手单位
        self.lot_size: int = int(config.get("lot_size", 100))

    def stop_distance(self, atr: float, atr_multiple: Optional[float] = None) -> float:
        """止损距离 = ATR × atr_multiple"""
        mult = self.atr_multiple if atr_multiple is None else atr_multiple
        return float(atr) * float(mult)

    def calculate(
        self,
        equity: float,
        entry_price: float,
        atr: float,
        atr_multiple: Optional[float] = None,
    ) -> int:
        """计算单只股票的目标买入股数

        参数：
            equity:       账户总权益
            entry_price:  入场价（当前收盘价）
            atr:          当前 ATR
            atr_multiple: 可选，覆盖默认止损倍数

        返回：
            目标股数（整手，向下取整）；无法计算时返回 0。
        """
        if equity <= 0 or entry_price <= 0 or atr <= 0:
            return 0

        stop_dist = self.stop_distance(atr, atr_multiple)
        if stop_dist <= 0:
            return 0

        # 风险预算法：风险金额 / 止损距离 = 股数
        risk_money = equity * self.risk_pct
        size = risk_money / stop_dist

        # 整手取整
        size = int(size / self.lot_size) * self.lot_size

        # 单股最大仓位封顶：max_position_pct × 资金 / 价格
        max_value = equity * self.max_position_pct
        max_size = int(max_value / entry_price / self.lot_size) * self.lot_size

        return max(0, min(size, max_size))

    def calculate_batch(
        self,
        equity: float,
        targets: Dict[str, Dict[str, float]],
        atr_map: Dict[str, float],
        atr_multiple: Optional[float] = None,
    ) -> Dict[str, int]:
        """批量计算目标股数

        参数：
            equity:   账户总权益
            targets:  {code: {"entry_price": float}} 结构
            atr_map:  {code: ATR}

        返回：
            {code: target_qty}；无法计算的 code 不出现在结果中。
        """
        out: Dict[str, int] = {}
        for code, t in targets.items():
            price = float(t["entry_price"])
            atr = float(atr_map.get(code, 0.0))
            qty = self.calculate(equity, price, atr, atr_multiple)
            if qty > 0:
                out[code] = qty
        return out

    def calculate_add(
        self,
        equity: float,
        current_qty: int,
        entry_price: float,
        atr: float,
        atr_multiple: Optional[float] = None,
    ) -> int:
        """计算加仓股数（目标仓位减去现有持仓）

        用于目标仓位 > 现有仓位时的补仓计算。返回 0 表示不需加仓。
        """
        target = self.calculate(equity, entry_price, atr, atr_multiple)
        delta = target - current_qty
        if delta <= 0:
            return 0
        return int(delta / self.lot_size) * self.lot_size

    def calculate_reduce(
        self,
        current_qty: int,
        target_qty: int,
    ) -> int:
        """计算减仓股数（现有持仓减去目标仓位）

        用于目标仓位 < 现有仓位时的减仓计算。返回 0 表示不需减仓。
        """
        delta = current_qty - target_qty
        if delta <= 0:
            return 0
        return int(delta / self.lot_size) * self.lot_size
