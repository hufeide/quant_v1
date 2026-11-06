"""竞技场策略基类：Backtrader Strategy 作为交易流程的协调者（Orchestrator）

本类不负责任任何具体计算，只负责按"先卖后买"的顺序协调各层模块：

    next():
        1. 卖出：risk_manager.check_all -> order_manager.sell -> portfolio.record_sell
        2. 买入：signal_ranker/signal_filter -> portfolio_manager.select_targets
               -> position_sizer.calculate -> risk_manager.check_buy
               -> order_manager.rebalance -> portfolio.record_buy
        3. 收盘：portfolio.mark_nav

职责分工（见用户架构说明）：
    signal_engine / signal_filter / signal_ranker：决定"买哪些"——信号层
    portfolio_manager：持仓数量、行业、总仓位——组合管理层（选股）
    position_sizer：每只股票买多少——组合管理层（仓位）
    risk_manager：风险预算、止损风险——组合管理层（风控）
    order_manager：目标仓位 -> 订单——执行层（唯一直接下单）
    trade_recorder：成交记录的统计——执行层（只读汇总）
"""
import backtrader as bt
from typing import Dict, List

from src.config import get_config, module_enabled
from src.portfolio.portfolio_manager import Portfolio
from src.portfolio.risk_manager import RiskManager
from src.portfolio.sentiment_gate import SentimentGate
from src.portfolio.position_sizer import PositionSizer
from src.execution.order_manager import OrderManager
from src.execution.trade_recorder import TradeRecorder
from src.backtest.broker import SimulatedBroker, Fill


class ArenaStrategyBase(bt.Strategy):
    """竞技场策略基类（协调者）"""

    params = dict(
        # 默认空 dict 而非 None：next() 里直接 .get(date)，传 None 会 AttributeError
        signals_dict={},
        sentiment_scores={},
        initial_cash=1_000_000,
        account_id="A1",
        detector_name="base",
    )

    def __init__(self):
        # 买入保护开关（消融配置在 config.yaml 的 universe.filters / risk / sentiment 段）
        self.buy_guards = {
            "no_chase_3pct": get_config("ablation.buy_guards.no_chase_3pct", True),
            "no_limit_up": get_config("ablation.buy_guards.no_limit_up", True),
            "pool_stale_halt": get_config("ablation.buy_guards.pool_stale_halt", True),
        }

        # ---------- 组合层：portfolio_manager（选股） ----------
        self.portfolio = Portfolio(
            account_id=self.p.account_id,
            cash=self.p.initial_cash,
            cfg={
                "cooldown_days": get_config("cooldown.days", 30),
                "max_buy_per_round": get_config("position.max_buy_per_round", 2),
                "max_buy_per_day": get_config("position.max_buy_per_day", 3),
                "max_holdings": get_config("position.max_holdings", 6),
                "target_weight": get_config("position.target_weight", 0.18),
                "buy_max_score": get_config("scoring.buy_max_score", 70.0),
                "portfolio_enabled": module_enabled("portfolio.enabled"),
                "holdings_limit": module_enabled("portfolio.holdings_limit"),
                "cooldown_enabled": module_enabled("portfolio.cooldown"),
                "nav_tracking": True,
                "score_filter": module_enabled("scoring.enabled"),
            }
        )

        # ---------- 执行层：broker + order_manager + trade_recorder ----------
        self.broker_sim = SimulatedBroker(
            self.p.initial_cash,
            get_config("broker", {})
        )
        self.order_manager = OrderManager(self.broker_sim, self, get_config("order", {}))
        self.trade_recorder = TradeRecorder(self.portfolio)

        # ---------- 组合层：position_sizer（仓位） ----------
        sizer_cfg = get_config("sizing", {})
        self.position_sizer = PositionSizer(sizer_cfg)

        # ---------- 组合层：risk_manager（风控，卖出 + 买入预算） ----------
        risk_cfg = get_config("risk", {})
        risk_cfg["total_risk_pct"] = sizer_cfg.get("total_risk_pct", 0.02)
        self.risk = RiskManager(risk_cfg)

        # ---------- 组合层：sentiment_gate（情绪闸门，只限买） ----------
        gate_cfg = dict(get_config("sentiment", {}))
        for k in ("gate_open", "gate_limit"):
            v = get_config(f"ablation.sentiment.{k}", None)
            if v is not None:
                gate_cfg[k] = v
        self.gate = SentimentGate(gate_cfg)

        # 当前日期
        self.current_date = None
        self.buy_count_today = 0

        # 数据映射：code -> bt.Data
        self.data_map: Dict[str, bt.Data] = {}
        for d in self.datas:
            if hasattr(d, "_name"):
                self.data_map[d._name] = d

        # MA 指标（B1 跌破均线判断）：backtrader 的 Data 上没有 data.ma20，
        # 必须显式创建并保存。
        self.ma_map: Dict[str, dict] = {}
        ma_period = int(get_config("risk.ma_exit", 20))
        for name, d in self.data_map.items():
            self.ma_map[name] = {
                ma_period: bt.ind.SMA(d.close, period=ma_period),
            }

    # ---------- 净值预热 ----------

    def prenext(self):
        """指标预热期补记净值（next() 尚未触发时）"""
        new_date = self.datas[0].datetime.date(0).isoformat()
        if new_date == self.current_date:
            return
        prices = {name: d.close[0] for name, d in self.data_map.items() if len(d)}
        self.portfolio.mark_nav(new_date, prices)
        self.current_date = new_date

    # ---------- 主流程 ----------

    def next(self):
        """主循环：先卖后买"""
        new_date = self.datas[0].datetime.date(0).isoformat()

        if new_date != self.current_date:
            self._on_date_change(new_date)
        self.current_date = new_date

        prices = {name: d.close[0] for name, d in self.data_map.items()}
        self.portfolio.mark_nav(self.current_date, prices)

        # 第一步：卖出（risk_manager 检查 -> order_manager 下单）
        self._execute_sells()

        # 第二步：买入（选股 -> 仓位 -> 风险 --> 下单）
        self._execute_buys()

    def _on_date_change(self, new_date: str):
        """日期变化时重置当日计数器"""
        self.portfolio.reset_daily_counters()
        self.order_manager.on_new_day(new_date)
        if hasattr(self, 'accounts') and self.accounts:
            for pf in self.accounts.values():
                pf.reset_daily_counters()

    # ---------- 卖出执行 ----------

    def _execute_sells(self, pf=None):
        """卖出：risk_manager 检查 -> order_manager 下单 -> portfolio 记账"""
        pf = pf if pf is not None else self.portfolio
        to_sell = []

        for code, pos in list(pf.holdings.items()):
            if code not in self.data_map:
                continue
            data = self.data_map[code]
            price = data.close[0]
            pre_close = data.close[-1] if len(data) > 1 else price
            limit_down = pre_close * (1.0 - self.broker_sim.limit_pct(code))

            pos.mark_to_market(price)
            pos.hold_days = self._hold_days(pos.buy_date, self.current_date)

            ma_val = self.ma_map.get(code, {}).get(20, [None])[0] if self.ma_map else None

            triggered = self.risk.check_all(
                pos, ma_value=ma_val, after_confirm_time=False,
                at_close=True, limit_down=limit_down, sealed=False,
            )
            if triggered:
                to_sell.append((code, pos, triggered))

        for code, pos, trig in to_sell:
            data = self.data_map[code]
            price = data.close[0]
            pre_close = data.close[-1] if len(data) > 1 else price
            limit_down = pre_close * (1.0 - self.broker_sim.limit_pct(code))
            fill = self.order_manager.sell(
                code, pos.qty, price, pre_close=pre_close,
                limit_down=limit_down, date=self.current_date,
            )
            if fill:
                pf.record_sell(fill, self.current_date, trig["reason"])

    # ---------- 买入执行 ----------

    def _execute_buys(self, pf=None, detector_name=None):
        """买入：情绪闸门 -> 选股 -> 仓位 -> 风险 --> 下单"""
        pf = pf if pf is not None else self.portfolio
        det_name = detector_name if detector_name is not None else self.p.detector_name

        # 1. 情绪闸门（只限买）
        sentiment = self.p.sentiment_scores.get(self.current_date)
        if module_enabled("sentiment.enabled"):
            gate_status = self.gate.check(sentiment)
            max_buy_round = gate_status["max_buy"]
            if gate_status["mode"] == "halted":
                return
        else:
            max_buy_round = int(get_config("ablation.position.max_buy_per_round", 2))

        # 2. 当日信号（signal_filter/ranker 的产物，已按 score 升序）
        signals_df = self.p.signals_dict.get(self.current_date)
        if signals_df is None or signals_df.empty:
            return
        my_signals = signals_df[signals_df["detector"] == det_name]
        if my_signals.empty:
            return
        my_signals = my_signals.sort_values("score")

        # 3. 选股：portfolio_manager.select_targets（选哪些）
        targets = pf.select_targets(
            my_signals, date=self.current_date, detector=det_name,
            round_budget=max_buy_round, max_rounds=max_buy_round,
        )
        if not targets:
            return

        # 4. 仓位：position_sizer.calculate（买多少）
        equity = pf.total_value(prices=self._current_prices())
        target_sizes: Dict[str, int] = {}
        sig_by_code = {t["code"]: t for t in targets}
        for t in targets:
            code = t["code"]
            entry_price = self._current_prices().get(code, float(t.get("entry_price", 0)))
            atr = float(t.get("atr", 0) or 0)
            qty = self.position_sizer.calculate(equity, entry_price, atr)
            if qty > 0:
                target_sizes[code] = qty
        if not target_sizes:
            return

        # 5. 风险：risk_manager.check_buy（买入侧风险预算）
        atr_map = {code: float(t.get("atr", 0) or 0) for code, t in sig_by_code.items()}
        check = self.risk.check_buy(target_sizes, pf.holdings, equity, atr_map)
        if not check.get("approved", True):
            # 记录拒绝原因（不阻断，回退到逐个下单保护）
            for k, v in check.get("risks", {}).items():
                pass  # 风险明细由 risk_manager 记录

        # 6. 下单：order_manager.rebalance（唯一直接下单层）
        def price_getter(code): return self._current_prices().get(code, 0.0)
        def pre_close_getter(code):
            d = self.data_map.get(code)
            if d and len(d) > 1:
                return d.close[-1]
            return 0.0
        def limit_down_getter(code):
            pc = pre_close_getter(code)
            return pc * (1.0 - self.broker_sim.limit_pct(code)) if pc else 0.0

        fills = self.order_manager.rebalance(
            target_sizes, current_sizes={},
            price_getter=price_getter,
            pre_close_getter=pre_close_getter,
            limit_down_getter=limit_down_getter,
            date=self.current_date, buy_guards=self.buy_guards,
        )

        # 7. 记账：portfolio.record_buy
        bought_this_round = 0
        for fill in fills:
            if bought_this_round >= max_buy_round:
                break
            pf.record_buy(fill, self.current_date, score=0.0)
            pf.buy_count_today += 1
            bought_this_round += 1

    def _current_prices(self):
        return {name: d.close[0] for name, d in self.data_map.items()}

    def _hold_days(self, buy_date: str, current_date: str) -> int:
        from datetime import date as _d
        da = _d.fromisoformat(buy_date[:10])
        db = _d.fromisoformat(current_date[:10])
        return (db - da).days

    # ---------- 收口清仓 ----------

    def _force_close(self, pf=None, reason="回测结束强制卖出"):
        """回测最后一天：按当日收盘价强制卖出全部剩余持仓"""
        pf = pf if pf is not None else self.portfolio
        if not pf.holdings:
            return
        date = self.current_date or (pf.daily_nav[-1]["date"] if pf.daily_nav else "")
        for code, pos in list(pf.holdings.items()):
            if code in self.data_map:
                data = self.data_map[code]
                price = data.close[0]
                pre_close = data.close[-1] if len(data) > 1 else price
            else:
                price = pos.current_price or pos.init_price
                pre_close = price
            limit_down = pre_close * (1.0 - self.broker_sim.limit_pct(code))
            fill = self.broker_sim.sell(
                code, price, pos.qty, pre_close=pre_close,
                limit_down=limit_down, ts=date,
            )
            if fill is None:
                fees = self.broker_sim.calc_fees(price, pos.qty, "sell")
                fill = Fill(code, "sell", price, pos.qty, fees["commission"],
                            fees["stamp_tax"], fees["transfer_fee"],
                            fees["total"], date)
            pf.record_sell(fill, date, reason)
        pf.mark_nav(date, {})

    def stop(self):
        """策略结束：强制清仓 + 输出统计"""
        self._force_close(self.portfolio)
        total_trades = len(self.portfolio.trades)
        buys = len([t for t in self.portfolio.trades if t["side"] == "buy"])
        sells = len([t for t in self.portfolio.trades if t["side"] == "sell"])
        final_nav = self.portfolio.daily_nav[-1]["nav"] if self.portfolio.daily_nav else 1.0

        print(f"[{self.p.account_id}] 策略结束:")
        print(f"  最终净值: {final_nav:.4f} ({(final_nav-1)*100:.2f}%)")
        print(f"  总交易: {total_trades} 笔 (买{buys} 卖{sells})")
        print(f"  持仓数: {len(self.portfolio.holdings)}")
