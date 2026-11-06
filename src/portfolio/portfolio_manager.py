"""单账户组合管理：持仓、冷却期、仓位约束、净值"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.portfolio.risk_manager import Position


class CooldownTracker:
    """卖出后 N 天内不买回（F: 冷却期）"""

    def __init__(self, days: int = 30):
        self.days = days
        self._sold: Dict[str, str] = {}   # code -> sell_date

    def mark_sold(self, code: str, date: str):
        self._sold[code] = date

    def clear_sold(self, code: str):
        """重新买入时清除冷却记录"""
        self._sold.pop(code, None)

    def in_cooldown(self, code: str, date: str) -> bool:
        if code not in self._sold:
            return False
        return self._date_diff(self._sold[code], date) < self.days

    @staticmethod
    def _date_diff(a: str, b: str) -> int:
        """交易日间隔，简单用日期字符串差（回测中按交易日历精化）"""
        from datetime import date as _d
        da = _d.fromisoformat(a[:10]) if isinstance(a, str) else a
        db = _d.fromisoformat(b[:10]) if isinstance(b, str) else b
        return (db - da).days


class Portfolio:
    """单账户组合"""

    def __init__(self, account_id: str, cash: float, cfg: Dict):
        self.account_id = account_id
        self.cfg = cfg
        self.broker_cash = cash
        self.initial_cash = float(cash)
        self.holdings: Dict[str, Position] = {}
        self.trades: List[Dict] = []
        self.cooldown = CooldownTracker(cfg.get("cooldown_days", 30))
        self.daily_nav: List[Dict] = []
        self.buy_count_today = 0
        self.max_buy_per_round = cfg.get("max_buy_per_round", 2)
        self.max_buy_per_day = cfg.get("max_buy_per_day", 3)
        self.max_holdings = cfg.get("max_holdings", 6)
        self.target_weight = cfg.get("target_weight", 0.18)
        self.buy_max_score = cfg.get("buy_max_score", 70.0)
        self.pool_stale = False   # F3：股票池落后时置 True
        self.buy_blocked_reason: str = ""

        # 组合管理开关（见 config.yaml modules.portfolio，由策略层注入）
        self.portfolio_enabled = cfg.get("portfolio_enabled", True)
        self.holdings_limit = cfg.get("holdings_limit", True)
        self.cooldown_enabled = cfg.get("cooldown_enabled", True)
        self.nav_tracking = cfg.get("nav_tracking", True)
        self.score_filter = cfg.get("score_filter", True)

    # ---------- 持仓 ----------

    def market_value(self, prices: Dict[str, float]) -> float:
        """当前持仓市值"""
        return sum(p.qty * prices.get(code, p.current_price) for code, p in self.holdings.items())

    def total_value(self, prices: Dict[str, float]) -> float:
        """总资产 = 现金 + 市值"""
        return self.broker_cash + self.market_value(prices)

    def used_weight(self) -> float:
        """已用仓位比例（相对总资产）"""
        return self.market_value({}) / max(self.total_value({}), 1e-9)

    # ---------- 目标选择（组合管理核心职责） ----------

    def select_targets(
        self,
        ranked_signals,
        *,
        date: str,
        detector: Optional[str] = None,
        round_budget: Optional[int] = None,
        max_rounds: Optional[int] = 1,
        industry_map: Optional[Dict[str, str]] = None,
        industry_max: int = 2,
    ) -> List[Dict]:
        """从"排序后的有效信号"中选出目标股票集合

        输入磐排后的信号（来自 signal_ranker），输出应买入的候选集合。
        本方法只负责"选哪些"，不决定"买多少"（position_sizer），
        不下单（order_manager）。

        决策逻辑：
            1. 按 detector 过滤（若指定）
            2. 通过 can_open 约束（持仓上限/当日买入/冷却/得分阈值）
            3. 逐轮递减轮次额度（round_budget），每轮取 max_rounds 只
            4. 行业集中度限制：新增买入后同行业敞口不超过 industry_max

        参数：
            ranked_signals: signal_ranker 的输出（DataFrame 或 record dict 列表）
            date:           交易日（用于冷却期判断）
            detector:       限定检测器名（配角策略用），None=不过滤
            round_budget:   本轮剩余买入额度（循环减到 0 即停）
            max_rounds:     本轮最多买几只（默认 1）
            industry_map:   {code: industry} 行业映射（无行业信息时传 None）
            industry_max:   同行业敞口上限

        返回：
            [{"code":..., "score":..., "entry_price":..., "atr":..., "detector":...}]
            已按 rank 顺序排列（rank 列存在时），否则按 score 升序。
        """
        candidates = self._to_records(ranked_signals)
        if detector:
            candidates = [c for c in candidates if c.get("detector") == detector]

        targets: List[Dict] = []
        budget = round_budget if round_budget is not None else max_rounds

        for c in candidates:
            if budget <= 0 and not self.portfolio_enabled:
                break
            if budget <= 0:
                break
            code = c["code"]
            ok, reason = self.can_open(code, date, float(c.get("score", 0.0)), budget)
            if not ok:
                continue
            # 行业集中度：加入该股后行业内敞口是否超限
            if industry_map is not None:
                indus = industry_map.get(code)
                if indus is not None and self.industry_count(indus) >= industry_max:
                    continue
            targets.append(c)
            budget -= 1

        return targets

    def industry_count(self, industry: str,
                       industry_map: Optional[Dict[str, str]] = None) -> int:
        """当前持仓中属于某行业的数量（行业集中度检查）"""
        if industry_map is None:
            return 0
        return sum(1 for c in self.holdings if industry_map.get(c) == industry)

    @staticmethod
    def _to_records(ranked_signals):
        """统一转成 dict 列表：支持 DataFrame / dict / list[dict]"""
        if ranked_signals is None:
            return []
        if isinstance(ranked_signals, dict):
            # 若是 {score: value} 这种展开式字典则先归一，否则视为单条记录
            if isinstance(ranked_signals.get("score"), (int, float)) \
                    and "code" in ranked_signals:
                return [ranked_signals]
            return list(ranked_signals.values())
        if isinstance(ranked_signals, list):
            return ranked_signals
        # DataFrame
        import pandas as pd
        if isinstance(ranked_signals, pd.DataFrame):
            if ranked_signals.empty:
                return []
            return ranked_signals.to_dict("records")
        return []

    def can_open(self, code: str, date: str, score: float,
                 round_budget: Optional[int] = None) -> tuple:
        """检查能否开新仓，返回 (ok, reason)

        约束开关（config.yaml modules.portfolio）：
          - portfolio_enabled：总开关，关闭时不做任何约束
          - holdings_limit：持仓只数上限 + 每日买入上限
          - cooldown_enabled：卖出冷却期
          - score_filter：buy_max_score 得分阈值
        股票池落后是数据质量问题，始终检查，不随开关关闭。
        """
        if self.pool_stale:
            return False, "股票池数据落后，停止买入(F3)"
        if code in self.holdings:
            return False, "已持有该股"
        if round_budget is not None and round_budget <= 0:
            return False, "本轮额度已用尽"

        if not self.portfolio_enabled:
            return True, "ok"   # 组合管理关闭：无持仓约束

        if self.holdings_limit:
            if len(self.holdings) >= self.max_holdings:
                return False, f"持仓已达上限 {self.max_holdings} 只"
            if self.buy_count_today >= self.max_buy_per_day:
                return False, f"当日买入已达上限 {self.max_buy_per_day} 只"
        if self.cooldown_enabled and self.cooldown.in_cooldown(code, date):
            return False, "冷却期内(卖出后30天不买回)"
        if self.score_filter and score >= self.buy_max_score:
            return False, f"综合得分 {score} 未低于 {self.buy_max_score}"
        return True, "ok"

    # ---------- 成交 ----------

    def record_buy(self, fill, date: str, score: float = 0.0):
        """记录买入成交"""
        price = fill.price
        qty = fill.qty
        pos = Position(
            code=fill.code, init_price=price, current_price=price,
            qty=qty, buy_date=date, max_price=price, peak_gain=0.0,
        )
        self.holdings[fill.code] = pos
        # 现金是唯一账簿：扣除成交额 + 费用
        self.broker_cash -= (fill.price * fill.qty + fill.total_cost)
        self.buy_count_today += 1
        # 注意：不清除冷却记录，卖出后 30 天不买回是硬约束
        self.trades.append({
            "account": self.account_id, "date": date, "code": fill.code,
            "side": "buy", "price": price, "qty": qty, "score": score,
            "cost": fill.total_cost, "reason": "",
        })

    def record_sell(self, fill, date: str, reason: str = ""):
        """记录卖出成交"""
        pos = self.holdings.pop(fill.code, None)
        # 现金增加：成交额 - 费用（卖出费用从回款中扣除）
        self.broker_cash += (fill.price * fill.qty - fill.total_cost)
        gain = (fill.price / pos.init_price - 1.0) if pos else 0.0
        held = (date[:10] >= pos.buy_date[:10]) if pos else 0
        if self.cooldown_enabled:
            self.cooldown.mark_sold(fill.code, date)
        self.trades.append({
            "account": self.account_id, "date": date, "code": fill.code,
            "side": "sell", "price": fill.price, "qty": fill.qty,
            "gain_pct": round(gain * 100, 2),
            "hold_days": self._hold_days(pos.buy_date if pos else "", date),
            "cost": fill.total_cost, "reason": reason,
        })

    @staticmethod
    def _hold_days(buy_date: str, sell_date: str) -> int:
        from datetime import date as _d
        if not buy_date:
            return 0
        da = _d.fromisoformat(buy_date[:10])
        db = _d.fromisoformat(sell_date[:10])
        return (db - da).days

    # ---------- 净值 ----------

    def mark_nav(self, date: str, prices: Dict[str, float]):
        """记录当日净值（开关：modules.portfolio.nav_tracking）"""
        # 更新持仓现价（风控判断需要，不受净值开关影响）
        for code, p in self.holdings.items():
            if code in prices:
                p.mark_to_market(prices[code])
                p.hold_days = self._hold_days(p.buy_date, date)
        if not self.nav_tracking:
            return
        rec = {
            "date": date,
            "account": self.account_id,
            "cash": round(self.broker_cash, 2),
            "market_value": round(self.market_value(prices), 2),
            "total": round(self.total_value(prices), 2),
            "nav": round(self.total_value(prices) / max(self.initial_cash, 1e-9), 6),
            "holdings": len(self.holdings),
        }
        # 同日重复调用（如策略 stop() 强制清仓后再 mark_nav）覆盖当日最后记录，
        # 避免同日产生两条净值（前一条持仓市值、后一条清仓现金）
        if self.daily_nav and self.daily_nav[-1]["date"] == rec["date"]:
            self.daily_nav[-1] = rec
        else:
            self.daily_nav.append(rec)

    def reset_daily_counters(self):
        """每日重置计数器（由调度层调用）"""
        self.buy_count_today = 0
