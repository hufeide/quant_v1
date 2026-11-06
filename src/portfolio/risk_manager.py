"""卖出风控：R1-R5 主规则 + B1-B3 兜底

每条规则返回 None（不触发）或结构化触发信息。
check_all 按优先级顺序扫描，返回优先级最高的一条触发项。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


PRIORITY = {
    "R4": 6,   # 硬止盈 18%
    "R3": 5,   # 移动止盈
    "R2": 4,   # 保盈止损
    "R1": 3,   # 止损 -6%
    "B1": 2,   # 跌破均线收盘确认
    "B2": 1,   # 时间止损
    "B3": 0,   # 跌停顺延
}


@dataclass
class Position:
    """持仓记录"""
    code: str
    init_price: float
    current_price: float
    qty: int
    buy_date: str
    hold_days: int = 0
    max_price: float = field(default=0.0)   # 持仓期最高价（移动止盈用）
    peak_gain: float = 0.0                  # 历史最高浮盈比例

    def mark_to_market(self, price: float):
        """按最新价更新持仓状态"""
        self.current_price = price
        if self.max_price <= 0:
            self.max_price = price
        self.max_price = max(self.max_price, price)
        self.peak_gain = max(self.peak_gain, price / self.init_price - 1.0)

    def gain(self) -> float:
        """当前浮盈比例"""
        return self.current_price / self.init_price - 1.0


class RiskManager:
    """卖出风控引擎"""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    # ---------- 单条规则 ----------

    def r1_stop_loss(self, pos: Position) -> Optional[Dict]:
        """R1 止损：浮亏 ≥ stop_loss"""
        thr = self.cfg.get("stop_loss", -6.0) / 100.0
        if pos.gain() <= thr:
            return {"rule": "R1", "reason": f"止损触发 浮亏{pos.gain()*100:.2f}%",
                    "priority": PRIORITY["R1"]}
        return None

    def r2_breakeven(self, pos: Position) -> Optional[Dict]:
        """R2 保盈止损

        breakeven_mode="profit": 浮盈曾达 trigger，回落至 exit 即卖
        breakeven_mode="cost":   回落至成本价（0%）即卖
        """
        trig = self.cfg.get("breakeven_trigger", 7.0) / 100.0
        mode = self.cfg.get("breakeven_mode", "profit")
        exit_thr = self.cfg.get("breakeven_exit", 3.0) / 100.0

        if pos.peak_gain < trig:
            return None  # 从未达到触发线

        target = exit_thr if mode == "profit" else 0.0
        if pos.gain() <= target:
            label = "保盈" if mode == "profit" else "回落成本"
            return {"rule": "R2", "reason": f"{label}止损触发 峰值{pos.peak_gain*100:.1f}% 现价{pos.gain()*100:.1f}%",
                    "priority": PRIORITY["R2"]}
        return None

    def r3_trailing(self, pos: Position) -> Optional[Dict]:
        """R3 移动止盈：浮盈 > trigger 后，自最高点回撤 ≥ dd"""
        trig = self.cfg.get("trailing_trigger", 12.0) / 100.0
        dd = self.cfg.get("trailing_drawdown", 5.0) / 100.0

        if pos.peak_gain <= trig:
            return None
        if pos.max_price <= 0:
            return None
        cur_dd = pos.current_price / pos.max_price - 1.0
        if cur_dd <= -dd:
            return {"rule": "R3", "reason": f"移动止盈 峰值{pos.peak_gain*100:.1f}% 回撤{cur_dd*100:.2f}%",
                    "priority": PRIORITY["R3"]}
        return None

    def r4_hard_profit(self, pos: Position) -> Optional[Dict]:
        """R4 硬止盈：浮盈 ≥ hard_profit"""
        thr = self.cfg.get("hard_profit", 18.0) / 100.0
        if pos.gain() >= thr:
            return {"rule": "R4", "reason": f"硬止盈 浮盈{pos.gain()*100:.2f}%",
                    "priority": PRIORITY["R4"]}
        return None

    def b1_ma_exit(self, pos: Position, ma_value: float, after_confirm_time: bool = False,
                   at_close: bool = False) -> Optional[Dict]:
        """B1 收盘前确认有效跌破均线（仅 14:50 后且临近收盘）"""
        if not after_confirm_time and not at_close:
            return None
        if pd_isna(ma_value) or ma_value <= 0:
            return None
        if pos.current_price < ma_value:
            return {"rule": "B1", "reason": f"有效跌破MA{self.cfg.get('ma_exit',20)}",
                    "priority": PRIORITY["B1"]}
        return None

    def b2_time_stop(self, pos: Position) -> Optional[Dict]:
        """B2 时间止损：持有满 N 天且收益不足"""
        days = self.cfg.get("time_stop_days", 20)
        min_gain = self.cfg.get("time_stop_min_gain", 9.0) / 100.0
        if pos.hold_days >= days and pos.gain() < min_gain:
            return {"rule": "B2", "reason": f"时间止损 持有{pos.hold_days}天 收益{pos.gain()*100:.2f}%",
                    "priority": PRIORITY["B2"]}
        return None

    def b3_limit_down(self, price: float, limit_down: float, sealed: bool = True) -> Optional[Dict]:
        """B3 跌停顺延：跌停封死不假装成交"""
        if not self.cfg.get("limit_down_defer", True):
            return None
        if sealed and price <= limit_down * 1.001:
            return {"rule": "B3", "reason": "跌停封死 顺延至开板", "priority": PRIORITY["B3"]}
        return None

    # ---------- 组合扫描 ----------

    # 每条规则对应一个消融开关，键名与 src/ablation/experiments.py 一致
    RULE_TOGGLE_KEYS = {
        "R1": "R1_stop_loss",
        "R2": "R2_breakeven",
        "R3": "R3_trailing",
        "R4": "R4_hard_profit",
        "B1": "B1_ma_exit",
        "B2": "B2_time_stop",
        "B3": "B3_limit_down",
    }

    def _rule_on(self, rule_id: str, default: bool = True) -> bool:
        """读取规则开关；未配置时按默认开启。

        兼容两个位置（先到先得）：
          ablation.risk.<KEY>       —— 消融框架运行时注入（旧）
          modules.risk.rules.<KEY>  —— 组合策略开关表（新，见 config.yaml modules.risk.rules）
        """
        from src.config import get_config, module_enabled
        toggle_key = self.RULE_TOGGLE_KEYS.get(rule_id, rule_id)
        legacy = get_config(f"ablation.risk.{toggle_key}", None)
        if legacy is not None:
            return bool(legacy)
        return module_enabled(f"risk.rules.{toggle_key}", default)

    def check_all(self, pos: Position, ma_value: float = None,
                  after_confirm_time: bool = False, at_close: bool = False,
                  limit_down: float = None, sealed: bool = False) -> Optional[Dict]:
        """按优先级返回唯一一条触发规则

        总开关 modules.risk.enabled（兼容 ablation.risk.enabled）：
        关闭后不做任何卖出检查，持仓不动。
        每条规则可由 modules.risk.rules.<RID> 单独关闭，用于测量该规则的贡献。
        关闭后该规则既不触发也不拦截——优先级排序只在启用的规则间进行。
        """
        from src.config import module_enabled
        if not module_enabled("risk.enabled"):
            return None

        checks = []
        if self._rule_on("R4"):
            checks.append(self.r4_hard_profit(pos))
        if self._rule_on("R3"):
            checks.append(self.r3_trailing(pos))
        if self._rule_on("R2"):
            checks.append(self.r2_breakeven(pos))
        if self._rule_on("R1"):
            checks.append(self.r1_stop_loss(pos))
        if self._rule_on("B1") and ma_value is not None:
            checks.append(self.b1_ma_exit(pos, ma_value, after_confirm_time, at_close))
        if self._rule_on("B2"):
            checks.append(self.b2_time_stop(pos))
        if self._rule_on("B3") and limit_down is not None:
            checks.append(self.b3_limit_down(pos.current_price, limit_down, sealed))

        triggered = [c for c in checks if c is not None]
        if not triggered:
            return None
        triggered.sort(key=lambda c: c["priority"], reverse=True)
        return triggered[0]

    # ---------- 买入侧风险检查（组合总风险/行业风险/回撤限制） ----------

    def check_buy(
        self,
        target_sizes: Dict[str, int],
        positions: Dict[str, "Position"],
        equity: float,
        atr_map: Dict[str, float],
        industry_map: Optional[Dict[str, str]] = None,
        industry_max: int = 2,
        atr_multiple: float = 2.0,
    ) -> Dict[str, Any]:
        """买入前的组合侧风险预算检查

        职责边界：本方法只回答"这些目标订单整体是否可执行"，不决定
        "买多少"（position_sizer 已算出），不下单（order_manager）。

        检查：
            1. 组合总风险上限：所有持仓 + 新目标的
               期望风险（qty × ATR × atr_multiple / equity）之和 <= total_risk_pct
            2. 行业敞口集中度：加入新目标后任一行业持仓只数 <= industry_max
            3. 回撤限制：若已设置 max_drawdown_limit 且当前回撤超过该值，拒绝全部新买入

        返回：
            {"approved": bool, "reason": str, "reject": bool,
             "risks": {code: risk_pct}, "industry_breach": bool}
        """
        total_risk_pct = float(self.cfg.get("total_risk_pct", 0.02))
        if not total_risk_pct or not equity > 0:
            return {"approved": True, "reason": "ok", "reject": False}

        total_risk = self._aggregate_risk(positions, equity, atr_map, target_sizes, atr_multiple)
        if total_risk_pct > 0 and total_risk > total_risk_pct:
            return {
                "approved": False, "reject": True,
                "reason": f"组合总风险 {total_risk:.4f} > 上限 {total_risk_pct:.4f}",
                "risks": self._per_code_risk(positions, equity, atr_map, target_sizes, atr_multiple),
            }

        # 行业敞口集中度
        if industry_map is not None:
            industry_counts: Dict[str, int] = {}
            for code in positions:
                ind = industry_map.get(code)
                if ind:
                    industry_counts[ind] = industry_counts.get(ind, 0) + 1
            for code in target_sizes:
                ind = industry_map.get(code)
                if ind and industry_counts.get(ind, 0) >= industry_max:
                    return {
                        "approved": False, "reject": True,
                        "reason": f"行业[{ind}]敞口将达上限 {industry_max} 只",
                        "risks": self._per_code_risk(positions, equity, atr_map, target_sizes, atr_multiple),
                    }
                ind2 = industry_map.get(code)
                if ind2:
                    industry_counts[ind2] = industry_counts.get(ind2, 0) + 1

        # 回撤限制（若配置了阈值）
        drawdown_cap = self.cfg.get("max_drawdown_limit")
        if drawdown_cap is not None:
            dd = self.cfg.get("current_drawdown")
            if dd is not None and dd <= drawdown_cap:
                return {
                    "approved": False, "reject": True,
                    "reason": f"账户回撤 {dd:.4f} <= 限制 {drawdown_cap:.4f}，暂停新买入",
                    "risks": self._per_code_risk(positions, equity, atr_map, target_sizes, atr_multiple),
                }

        return {
            "approved": True, "reason": "ok", "reject": False,
            "risks": self._per_code_risk(positions, equity, atr_map, target_sizes, atr_multiple),
        }

    def _aggregate_risk(self, positions, equity, atr_map, target_sizes, atr_multiple):
        """持仓 + 目标订单的期望风险比例之和

        risk_pct = qty × ATR × atr_multiple / equity
        """
        total = 0.0
        for code, pos in positions.items():
            atr = atr_map.get(code, 0.0)
            total += pos.qty * atr * atr_multiple / equity
        for code, qty in target_sizes.items():
            atr = atr_map.get(code, 0.0)
            total += qty * atr * atr_multiple / equity
        return total

    def _per_code_risk(self, positions, equity, atr_map, target_sizes, atr_multiple):
        risks = {}
        for code in list(positions) + list(target_sizes):
            qty = positions[code].qty if code in positions else target_sizes.get(code, 0)
            atr = atr_map.get(code, 0.0)
            risks[code] = qty * atr * atr_multiple / equity if equity > 0 else 0.0
        return risks


def pd_isna(v) -> bool:
    """避免 pandas 依赖的空值判断"""
    try:
        return v != v
    except Exception:
        return False
