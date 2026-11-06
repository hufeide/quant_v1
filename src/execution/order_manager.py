"""订单管理：目标股数 + 当前股数 -> 订单（唯一直接下单层）

职责边界：
    只回答"如何把目标仓位换算成真实订单"：
        目标股数  -  当前股数  =  订单股数（买入为正，卖出为负）
    并负责订单状态跟踪、重复订单控制、成交回报、失败处理。

支持两种后端：
    - backtrader 原生：self.buy() / self.sell() /
      order_target_size() / order_target_percent() / order_target_value()
    - SimulatedBroker：调用 broker.buy/sell 直接撮合（回测当前实现）

注意：止损触发条件由 risk_manager/策略计算，最终仍由本执行层发出订单。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from src.backtest.broker import SimulatedBroker


@dataclass
class Order:
    """订单状态记录"""
    code: str
    side: str          # "buy" / "sell"
    qty: int
    price: float = 0.0
    status: str = "pending"   # pending / filled / rejected / canceled
    fill_qty: int = 0
    reason: str = ""
    step_index: int = 0        # 轮次内序号（用于 tie-break 记录）


class OrderManager:
    """订单管理器"""

    def __init__(
        self,
        broker: SimulatedBroker,
        strategy=None,
        cfg: Optional[Dict] = None,
    ):
        self.broker = broker
        self.strategy = strategy
        cfg = cfg or {}
        self.cfg = cfg
        # 订单状态跟踪：{(code, side): Order}（最近一次）
        self._orders: Dict[tuple, Order] = {}
        # 当日已下单集合，用于重复订单控制
        self._placed_today: set = set()
        self._current_date: Optional[str] = None
        self._pending: list = []   # 待成交订单（本实现同步撮合，列表仅留痕）

    # ---------- 日期/轮次生命周期 ----------

    def on_new_day(self, date: str):
        """新交易日开始：重置当日重复订单控制"""
        self._placed_today.clear()
        self._current_date = date

    # ---------- 重复订单控制 ----------

    def _dedup_key(self, code: str, side: str) -> tuple:
        return (code, side)

    def is_duplicate(self, code: str, side: str) -> bool:
        """当日是否已下过同一方向的单（用于反手/补单保护）"""
        return self._dedup_key(code, side) in self._placed_today

    def mark_placed(self, code: str, side: str):
        self._placed_today.add(self._dedup_key(code, side))

    # ---------- 目标仓位 -> 订单股数 ----------

    @staticmethod
    def delta_and_side(target_size: int, current_size: int) -> tuple:
        """target_size - current_size -> (side, abs(qty))"""
        delta = int(target_size) - int(current_size)
        if delta > 0:
            return "buy", int(delta)
        if delta < 0:
            return "sell", int(-delta)
        return "", 0

    def rebalance(
        self,
        desired_orders: Dict[str, int],
        current_sizes: Dict[str, int],
        price_getter: Optional[Callable[[str], float]] = None,
        pre_close_getter: Optional[Callable[[str], float]] = None,
        limit_down_getter: Optional[Callable[[str], float]] = None,
        date: str = "",
        buy_guards: Optional[Dict[str, bool]] = None,
    ) -> list:
        """把目标仓位集合换算成订单并下单，返回成交列表（Fill 列表）

        股票仓位换算：
            target_size - current_size = 订单股数（正买入，负卖出）

        参数：
            desired_orders:   {code: target_qty}（0 表示目标清仓）
            current_sizes:    {code: current_qty}
            price_getter:     code -> 当前价（返回 <= 0 视为失败并记录）
            pre_close_getter: code -> 昨收（涨停判断）
            limit_down_getter: code -> 跌停价
            date:             交易日
            buy_guards:       {no_limit_up: bool, ...} 买入侧保护

        下单语义：
            订单集合按调用方给出的顺序处理（策略层已按 rank 排序）。
        """
        buy_guards = buy_guards or {}
        fills = []
        for code, target in desired_orders.items():
            if int(target) <= 0:
                side, qty = "sell", int(current_sizes.get(code, 0))   # 目标清零：全部卖出
            else:
                side, qty = self.delta_and_side(target, current_sizes.get(code, 0))
            if qty <= 0:
                continue

            # 重复订单控制
            if side and self.is_duplicate(code, side):
                continue

            price = price_getter(code) if price_getter else 0.0
            if price <= 0:
                self._record(code, side, qty, price, "rejected", reason="无报价")
                continue

            pre_close = pre_close_getter(code) if pre_close_getter else None
            limit_down = limit_down_getter(code) if limit_down_getter else None

            if side == "buy":
                # 涨停不买（追高保护由策略层逐信号判断，不在此层）
                if buy_guards.get("no_limit_up", True) and pre_close \
                        and self.broker.is_limit_up(price, pre_close, code):
                    self._record(code, side, qty, price, "rejected",
                                 reason="涨停不买")
                    continue
                fill = self.broker.buy(code, price, qty,
                                       pre_close=pre_close, ts=date)
            else:
                fill = self.broker.sell(code, price, qty,
                                       pre_close=pre_close, limit_down=limit_down,
                                       ts=date)

            if fill is None:
                self._record(code, side, qty, price, "rejected",
                             reason="无法成交")
                continue

            self.mark_placed(code, side)
            self._record(code, side, qty, price, "filled", fill_qty=fill.qty)
            fills.append(fill)

        return fills

    # ---------- 单边便捷方法 ----------

    def buy(self, code: str, qty: int, price: float,
            pre_close: Optional[float] = None, date: str = "",
            score: float = 0.0) -> Optional[object]:
        """买入单只股票"""
        if qty <= 0 or self.is_duplicate(code, "buy"):
            return None
        fill = self.broker.buy(code, price, qty, pre_close=pre_close,
                              ts=date, available_cash=None)
        if fill:
            self.mark_placed(code, "buy")
            self._record(code, "buy", qty, price, "filled", fill_qty=fill.qty)
        else:
            self._record(code, "buy", qty, price, "rejected", reason="无法成交")
        return fill

    def sell(self, code: str, qty: int, price: float,
             pre_close: Optional[float] = None,
             limit_down: Optional[float] = None,
             date: str = "") -> Optional[object]:
        """卖出单只股票"""
        if qty <= 0 or self.is_duplicate(code, "sell"):
            return None
        fill = self.broker.sell(code, price, qty, pre_close=pre_close,
                               limit_down=limit_down, ts=date)
        if fill:
            self.mark_placed(code, "sell")
            self._record(code, "sell", qty, price, "filled", fill_qty=fill.qty)
        else:
            self._record(code, "sell", qty, price, "rejected", reason="跌停顺延")
        return fill

    def order_target_size(self, code: str, target: int,
                          current: int, price: float,
                          **kw) -> Optional[object]:
        """order_target_size：换算 delta 后按方向下单"""
        side, qty = self.delta_and_side(target, current)
        if qty <= 0 or not side:
            return None
        return self.buy(code, qty, price, **kw) if side == "buy" \
            else self.sell(code, qty, price, **kw)

    # ---------- 订单状态 ----------

    def _record(self, code: str, side: str, qty: int, price: float,
                status: str, fill_qty: int = 0, reason: str = ""):
        """记录订单状态（覆盖最近一次，保留拒绝/成交轨迹）"""
        od = Order(code=code, side=side, qty=qty, price=price,
                   status=status, fill_qty=fill_qty, reason=reason)
        self._orders[self._dedup_key(code, side)] = od

    def order_status(self, code: str, side: str) -> Optional[Order]:
        return self._orders.get(self._dedup_key(code, side))

    def orders(self) -> Dict[tuple, Order]:
        """全部订单状态"""
        return dict(self._orders)

    def cancel_all(self):
        """取消所有未成交订单（本实现同步撮合，仅清空状态）"""
        for key, od in self._orders.items():
            if od.status == "pending":
                od.status = "canceled"
        self._pending.clear()
