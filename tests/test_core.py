#!/usr/bin/env python3
"""单元测试：风控规则、情绪闸门、撮合费用、冷却期

用法： python tests/test_core.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio.risk_manager import RiskManager, Position
from src.portfolio.sentiment_gate import SentimentGate
from src.backtest.broker import SimulatedBroker
from src.portfolio.portfolio_manager import Portfolio, CooldownTracker


PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


# ==================== 风控规则 ====================
print("== 风控规则 ==")
rm = RiskManager({
    "stop_loss": -6.0, "breakeven_trigger": 7.0, "breakeven_exit": 3.0,
    "breakeven_mode": "profit", "trailing_trigger": 12.0, "trailing_drawdown": 5.0,
    "hard_profit": 18.0, "ma_exit": 20, "time_stop_days": 20, "time_stop_min_gain": 9.0,
    "limit_down_defer": True,
})

# R1 止损
p = Position("600000", 10.0, 9.30, 1000, "2025-09-01")   # 亏 7%
r = rm.check_all(p)
check("R1 止损触发", r and r["rule"] == "R1", str(r))

p2 = Position("600000", 10.0, 9.50, 1000, "2025-09-01")   # 亏 5%
r2 = rm.check_all(p2)
check("R1 不触发(亏5%)", r2 is None, str(r2))

# R2 保盈
p3 = Position("600001", 10.0, 10.03, 1000, "2025-09-01", peak_gain=0.071)
r3 = rm.check_all(p3)
check("R2 保盈触发", r3 and r3["rule"] == "R2", str(r3))

p4 = Position("600002", 10.0, 10.05, 1000, "2025-09-01", peak_gain=0.071)
# 10.05 的浮盈只有 0.5%，已经跌破 3% 触发线，R2 应触发
r4 = rm.check_all(p4)
check("R2 回落到3%以下触发", r4 and r4["rule"] == "R2", str(r4))

# 现价 10.5 浮盈 5%，高于 3% 线，R2 不触发
p4b = Position("600002", 10.0, 10.5, 1000, "2025-09-01", peak_gain=0.071)
r4b = rm.check_all(p4b)
check("R2 未回落不触发", r4b is None, str(r4b))

# R3 移动止盈：浮盈 12%~18% 之间（避开 R4 的 18% 阈值）
# 现价 11.50 → 浮盈 15%，低于 18%，走 R3 分支
p5 = Position("600003", 10.0, 11.50, 1000, "2025-09-01", max_price=12.13, peak_gain=0.15)
r5 = rm.check_all(p5)
check("R3 移动止盈触发", r5 and r5["rule"] == "R3", str(r5))

# 现价 11.70，回撤只有 3.5% < 5%，R3 不触发
p6 = Position("600004", 10.0, 11.70, 1000, "2025-09-01", max_price=12.13, peak_gain=0.15)
r6 = rm.check_all(p6)
check("R3 未回撤5%不触发", r6 is None, str(r6))

# R4 硬止盈
p7 = Position("600005", 10.0, 11.80, 1000, "2025-09-01", peak_gain=0.18)
r7 = rm.check_all(p7)
check("R4 硬止盈触发", r7 and r7["rule"] == "R4", str(r7))

# B2 时间止损
p8 = Position("600006", 10.0, 10.2, 1000, "2025-08-01", hold_days=21, peak_gain=0.05)
r8 = rm.check_all(p8)
check("B2 时间止损触发", r8 and r8["rule"] == "B2", str(r8))

# 优先级：R4 > R3 > R2 > R1
p9 = Position("600007", 10.0, 11.80, 1000, "2025-09-01",
              max_price=13.0, peak_gain=0.30)
r9 = rm.check_all(p9)
check("优先级 R4>R3", r9 and r9["rule"] == "R4", str(r9))


# ==================== 情绪闸门 ====================
print("\n== 情绪闸门 ==")
gate = SentimentGate({
    "weights": {"profit_effect": 0.45, "market_temp": 0.35, "retail_sentiment": 0.20},
    "gate_open": 52.0, "gate_limit": 40.0, "fail_open": True,
    "max_buy_per_round_default": 2,
})

score_hi = gate.combine(60, 60, 60)
check("合成分计算", abs(score_hi - 60.0) < 0.01, str(score_hi))

c = gate.check(60.0)
check(">52 正常开仓", c["mode"] == "open" and c["max_buy"] == 2, str(c))

c = gate.check(50.0)
check("40-52 限买1只", c["mode"] == "limited" and c["max_buy"] == 1, str(c))

c = gate.check(30.0)
check("<=40 暂停开仓", c["mode"] == "halted" and c["max_buy"] == 0, str(c))

c = gate.check(None)
check("缺失数据 fail-open", c["mode"] == "unknown" and c["max_buy"] == 2, str(c))


# ==================== 撮合费用 ====================
print("\n== 撮合费用 ==")
br = SimulatedBroker(1_000_000, {
    "commission_rate": 0.00025, "commission_min": 5.0,
    "stamp_tax_rate": 0.001, "transfer_fee_rate": 0.00001,
})

f = br.calc_fees(10.0, 10000, "buy")
turnover = 100000.0
expected = max(turnover * 0.00025, 5.0) + 0 + turnover * 0.00001
check("买入费用", abs(f["total"] - expected) < 0.01, f"{f['total']} vs {expected}")

f = br.calc_fees(10.0, 10000, "sell")
expected_s = max(turnover * 0.00025, 5.0) + turnover * 0.001 + turnover * 0.00001
check("卖出含印花税", abs(f["total"] - expected_s) < 0.01, f"{f['total']} vs {expected_s}")

check("涨停不买", br.is_limit_up(11.01, 10.0))
check("非涨停可买", not br.is_limit_up(10.5, 10.0))
check("跌停封死", br.is_limit_down_sealed(9.0, 10.0))
check("非跌停", not br.is_limit_down_sealed(9.5, 10.0))

fill = br.buy("600000", 10.0, 1000, available_cash=50000)
check("买入下单", fill is not None and fill.qty == 1000)
fill_fail = br.buy("600000", 10.0, 1000, available_cash=500)
check("现金不足拒单", fill_fail is None)


# ==================== 冷却期 ====================
print("\n== 冷却期 ==")
cd = CooldownTracker(days=30)
cd.mark_sold("600000", "2025-09-01")
check("30天内不可买回", cd.in_cooldown("600000", "2025-09-20"))
check("30天后可买回", not cd.in_cooldown("600000", "2025-10-15"))
check("未卖出无冷却", not cd.in_cooldown("600001", "2025-09-20"))


# ==================== Portfolio ====================
print("\n== Portfolio ==")
pf = Portfolio("A1", 1_000_000, {
    "cooldown_days": 30, "max_buy_per_round": 2, "max_buy_per_day": 3,
    "max_holdings": 6, "target_weight": 0.18, "buy_max_score": 70.0,
})

ok, reason = pf.can_open("600000", "2025-09-01", score=65.0)
check("得分<70 可买", ok, reason)

ok, reason = pf.can_open("600000", "2025-09-01", score=75.0)
check("得分>=70 拒买", not ok, reason)

# 持仓上限
for i in range(6):
    pf.holdings[f"60000{i}"] = Position(f"60000{i}", 10.0, 10.0, 100, "2025-09-01")
ok, reason = pf.can_open("600009", "2025-09-01", score=60.0)
check("持仓满6只拒买", not ok and "上限" in reason, reason)

# F3 股票池落后
pf.pool_stale = True
ok, reason = pf.can_open("600001", "2025-09-01", score=60.0)
check("池落后停买(F3)", not ok and "落后" in reason, reason)

# 现金账簿一致性
pf2 = Portfolio("A1", 1_000_000, {"cooldown_days": 30, "buy_max_score": 70.0,
                                  "max_buy_per_day": 3, "max_holdings": 6})
from src.backtest.broker import Fill
buy_fill = Fill("600000", "buy", 10.0, 10000, 25.0, 0.0, 1.0, 26.0, "2025-09-01")
pf2.record_buy(buy_fill, "2025-09-01", 60.0)
check("买入扣现金", abs(pf2.broker_cash - (1_000_000 - 100000 - 26.0)) < 0.01,
      str(pf2.broker_cash))

sell_fill = Fill("600000", "sell", 11.0, 10000, 27.5, 110.0, 1.1, 138.6, "2025-09-05")
pf2.record_sell(sell_fill, "2025-09-05", "R4")
check("卖出回现金", abs(pf2.broker_cash - (1_000_000 - 100000 - 26.0 + 110000 - 138.6)) < 0.01,
      str(pf2.broker_cash))
check("卖出后冷却", pf2.cooldown.in_cooldown("600000", "2025-09-20"))


# ==================== 组合策略开关表（config.yaml modules 段） ====================
print("== 组合策略开关表 ==")
from src.config import module_enabled, set_override, clear_override

# 默认全部启用（config.yaml modules 段声明）
check("modules.detectors.enabled 默认开启", module_enabled("detectors.enabled") is True)
check("modules.risk.enabled 默认开启", module_enabled("risk.enabled") is True)
check("modules.sentiment.enabled 默认开启", module_enabled("sentiment.enabled") is True)
check("modules.portfolio.enabled 默认开启", module_enabled("portfolio.enabled") is True)
check("modules.scoring.enabled 默认开启", module_enabled("scoring.enabled") is True)
check("逐条规则默认开启", module_enabled("risk.rules.B1_ma_exit") is True)

# 消融覆盖（ablation.*）优先于 config.yaml 的 modules.*
set_override({"ablation": {"sentiment": {"enabled": False}}})
check("ablation 覆盖优先于 modules 表", module_enabled("sentiment.enabled") is False)
clear_override()

# ---------- 风控总开关 ----------
set_override({"modules": {"risk": {"enabled": False}}})
rm_off = RiskManager({"stop_loss": -6.0})
p_off = Position("600000", 10.0, 9.30, 1000, "2025-09-01")   # 亏 7%，正常应触发 R1
check("风控总开关关闭：R1 不触发", rm_off.check_all(p_off) is None)
clear_override()
rm_on = RiskManager({"stop_loss": -6.0})
r_on = rm_on.check_all(p_off)
check("风控总开关恢复：R1 触发", r_on is not None and r_on["rule"] == "R1", str(r_on))

# ---------- 逐条规则开关 ----------
set_override({"modules": {"risk": {"rules": {"R1_stop_loss": False}}}})
rm_r1off = RiskManager({"stop_loss": -6.0})
check("关闭 R1 后该规则不触发", rm_r1off.check_all(p_off) is None)
clear_override()

# ---------- 组合管理开关 ----------
pf_off = Portfolio("A1", 1_000_000, {
    "cooldown_days": 30, "buy_max_score": 70.0, "max_buy_per_day": 3, "max_holdings": 6,
    "portfolio_enabled": False, "holdings_limit": False, "cooldown_enabled": False,
    "nav_tracking": False, "score_filter": False,
})
check("组合管理关闭：高分行可买", pf_off.can_open("600100", "2025-09-01", 95.0)[0])
for i in range(6):
    pf_off.holdings[f"60000{i}"] = Position(f"60000{i}", 10.0, 10.0, 100, "2025-09-01")
check("组合管理关闭：持仓超限仍可买", pf_off.can_open("600009", "2025-09-01", 95.0)[0])

# 单项开关：持仓上限
pf_hl = Portfolio("A1", 1_000_000, {
    "cooldown_days": 30, "buy_max_score": 70.0, "max_buy_per_day": 3, "max_holdings": 6,
    "holdings_limit": False,
})
for i in range(6):
    pf_hl.holdings[f"60000{i}"] = Position(f"60000{i}", 10.0, 10.0, 100, "2025-09-01")
check("关闭持仓上限：可继续买", pf_hl.can_open("600009", "2025-09-01", 60.0)[0])

# 单项开关：冷却期
pf_cd = Portfolio("A1", 1_000_000, {
    "cooldown_days": 30, "buy_max_score": 70.0, "max_buy_per_day": 3, "max_holdings": 6,
    "cooldown_enabled": False,
})
pf_cd.cooldown.mark_sold("600000", "2025-08-01")
check("关闭冷却期：期内可买回", pf_cd.can_open("600000", "2025-08-10", 60.0)[0])

# 单项开关：净值追踪
pf_nav = Portfolio("A1", 1_000_000, {
    "cooldown_days": 30, "buy_max_score": 70.0, "max_buy_per_day": 3, "max_holdings": 6,
    "nav_tracking": False,
})
pf_nav.mark_nav("2025-09-01", {})
check("关闭净值追踪：无 daily_nav", pf_nav.daily_nav == [])

# ---------- 检测器开关 ----------
from src.signals.signal_engine import build_detectors
from src.config import load_config
cfg = load_config()
dets_all = build_detectors(cfg)
check("默认构建四个检测器", len(dets_all) == 4, str(list(dets_all)))
set_override({"modules": {"detectors": {"enabled": False}}})
check("检测器总开关关闭：无检测器", build_detectors(cfg) == {})
clear_override()
set_override({"modules": {"detectors": {"pullback": False}}})
dets_off = build_detectors(cfg)
check("分开关关闭 pullback：其余保留",
      "pullback" not in dets_off and len(dets_off) == 3, str(list(dets_off)))
clear_override()


print(f"\n{'='*40}\n通过 {PASS} 项，失败 {FAIL} 项\n{'='*40}")
sys.exit(1 if FAIL else 0)
