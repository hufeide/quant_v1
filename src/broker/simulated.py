"""模拟撮合：佣金/印花税/过户费 + 跌停不成交

不依赖 backtrader broker，可独立单元测试，
也被 backtrader 策略直接调用。
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Fill:
    """成交记录"""
    code: str
    side: str            # "buy" / "sell"
    price: float
    qty: int
    commission: float
    stamp_tax: float
    transfer_fee: float
    total_cost: float
    ts: str = ""


class SimulatedBroker:
    """A 股模拟撮合"""

    def __init__(self, cash: float, cfg: Dict):
        # cash 仅用于下单前的可用性校验；账簿现金由 Portfolio 持有，
        # 避免 broker 与 portfolio 双重记账导致现金被扣两次。
        self.cash = float(cash)
        self.cfg = cfg

    # ---------- 费用 ----------

    def commission(self, turnover: float) -> float:
        """佣金：万 2.5，最低 5 元"""
        rate = self.cfg.get("commission_rate", 0.00025)
        floor = self.cfg.get("commission_min", 5.0)
        return max(turnover * rate, floor)

    def stamp_tax(self, turnover: float, side: str) -> float:
        """印花税：仅卖出，千分之一"""
        if side != "sell":
            return 0.0
        return turnover * self.cfg.get("stamp_tax_rate", 0.001)

    def transfer_fee(self, turnover: float) -> float:
        """过户费：万 0.1"""
        return turnover * self.cfg.get("transfer_fee_rate", 0.00001)

    def calc_fees(self, price: float, qty: int, side: str) -> Dict[str, float]:
        """费用明细"""
        turnover = price * qty
        c = self.commission(turnover)
        s = self.stamp_tax(turnover, side)
        t = self.transfer_fee(turnover)
        return {"commission": c, "stamp_tax": s, "transfer_fee": t,
                "total": c + s + t, "turnover": turnover}

    # ---------- 价格有效性 ----------

    def is_limit_down_sealed(self, price: float, pre_close: float,
                             limit_down: Optional[float] = None) -> bool:
        """跌停封死判断（F2：跌停卖不出就是卖不出）"""
        ld = limit_down if limit_down is not None else pre_close * 0.90
        return price <= ld * 1.001

    def is_limit_up(self, price: float, pre_close: float) -> bool:
        """涨停判断：涨停不买"""
        return price >= pre_close * 1.10 * 0.999

    def price_is_stale(self, ts_epoch: float, now_epoch: float) -> bool:
        """F1：价格过期判断"""
        max_age = self.cfg.get("price_stale_seconds", 30)
        return (now_epoch - ts_epoch) > max_age

    # ---------- 下单 ----------

    def can_buy(self, price: float, qty: int, pre_close: float = None,
                available_cash: float = None) -> bool:
        """买入可行性：现金够 + 非涨停 + 整手

        available_cash 由调用方（Portfolio）传入其实时现金余额；
        未传入时回退到构造时记录的初始现金。
        """
        if qty <= 0 or qty % 100 != 0:
            return False
        fees = self.calc_fees(price, qty, "buy")
        cash = available_cash if available_cash is not None else self.cash
        if fees["turnover"] + fees["total"] > cash:
            return False
        if pre_close and self.is_limit_up(price, pre_close):
            return False
        return True

    def buy(self, code: str, price: float, qty: int,
            pre_close: float = None, ts: str = "",
            available_cash: float = None) -> Optional[Fill]:
        """买入：返回成交记录或 None（不改动任何账簿现金）"""
        if not self.can_buy(price, qty, pre_close, available_cash):
            return None
        fees = self.calc_fees(price, qty, "buy")
        return Fill(code, "buy", price, qty, fees["commission"],
                    fees["stamp_tax"], fees["transfer_fee"],
                    fees["total"], ts)

    def sell(self, code: str, price: float, qty: int,
             pre_close: float = None, limit_down: float = None,
             ts: str = "") -> Optional[Fill]:
        """卖出：跌停封死则不成交（F2）"""
        if qty <= 0 or qty % 100 != 0:
            return None
        if pre_close is not None and self.is_limit_down_sealed(price, pre_close, limit_down):
            return None
        fees = self.calc_fees(price, qty, "sell")
        return Fill(code, "sell", price, qty, fees["commission"],
                    fees["stamp_tax"], fees["transfer_fee"],
                    fees["total"], ts)
