"""竞技场策略基类：六账户统一风控 + 先卖后买

设计要点：
  1. 每个策略维护独立 Portfolio（模拟账户）
  2. next() 中先执行卖出（风险检查），再执行买入（情绪闸门 + 仓位约束）
  3. 情绪闸门只限买，不影响卖出
  4. 跌停封死时不成交，顺延到次日
"""
import backtrader as bt
from typing import Dict, List
from src.portfolio import Portfolio
from src.broker.simulated import SimulatedBroker, Fill
from src.risk.rules import RiskManager
from src.risk.sentiment_gate import SentimentGate
from src.config import load_config, get_config, module_enabled


class ArenaStrategyBase(bt.Strategy):
    """竞技场策略基类"""

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
                # 组合管理开关（config.yaml modules.portfolio / modules.scoring）
                "portfolio_enabled": module_enabled("portfolio.enabled"),
                "holdings_limit": module_enabled("portfolio.holdings_limit"),
                "cooldown_enabled": module_enabled("portfolio.cooldown"),
                "nav_tracking": True,   # 净值追踪默认打开，不再受模块开关影响
                "score_filter": module_enabled("scoring.enabled"),
            }
        )
        self.broker_sim = SimulatedBroker(
            self.p.initial_cash,
            get_config("broker", {})
        )
        self.risk = RiskManager(get_config("risk", {}))
        # 情绪闸门：叠加消融对阈值的覆盖（ablation.sentiment.gate_open/gate_limit）
        gate_cfg = dict(get_config("sentiment", {}))
        for k in ("gate_open", "gate_limit"):
            v = get_config(f"ablation.sentiment.{k}", None)
            if v is not None:
                gate_cfg[k] = v
        self.gate = SentimentGate(gate_cfg)

        # 当前日期字符串
        self.current_date = None
        self.buy_count_today = 0

        # 数据映射：code -> bt.Data
        self.data_map: Dict[str, bt.Data] = {}
        for d in self.datas:
            if hasattr(d, "_name"):
                self.data_map[d._name] = d

        # 为每只股票的 data feed 建 MA 指标。
        # backtrader 的 Data 对象上没有 data.ma20 这种属性，
        # 必须用 bt.ind.SMA(d, period=20) 显式创建并自己保存。
        self.ma_map: Dict[str, dict] = {}
        ma_period = int(get_config("risk.ma_exit", 20))
        for name, d in self.data_map.items():
            self.ma_map[name] = {
                ma_period: bt.ind.SMA(d.close, period=ma_period),
            }

        # 收集数据用于指标计算（如果策略需要）
        self.indicators = {}

    def detect_pattern(self, data: bt.Data, dt: int) -> bool:
        """子类实现：检测形态，返回 bool"""
        return False

    def prenext(self):
        """指标预热期（SMA 等指标未就绪、next() 尚未被调用）补记净值

        backtrader 在指标 minperiod 达成前只调用 prenext 而不会调用 next，
        此前净值曲线缺失行情首日后的 minperiod-1 个交易日（约 19 天）。
        本保单只做净值时间序列，不动买卖与当日计数器（_on_date_change
        仍由 next() 在真正开始交易时统一处理）。
        """
        new_date = self.datas[0].datetime.date(0).isoformat()
        if new_date == self.current_date:
            return
        prices = {name: d.close[0] for name, d in self.data_map.items() if len(d)}
        self.portfolio.mark_nav(new_date, prices)
        self.current_date = new_date

    def next(self):
        """主循环：先卖后买"""
        # 获取当前日期
        new_date = self.datas[0].datetime.date(0).isoformat()

        # 日期变化时重置所有账户的当日计数（第 10.1 节修复：
        # 原来在 next() 末尾无条件重置 self.buy_count_today，但
        # can_open() 用的是 pf.buy_count_today，从不重置 →
        # max_buy_per_day 退化为全回测上限）
        if new_date != self.current_date:
            self._on_date_change(new_date)
        self.current_date = new_date

        # 收盘净值记录
        prices = {name: d.close[0] for name, d in self.data_map.items()}
        self.portfolio.mark_nav(self.current_date, prices)

        # ========== 第一步：卖出（风控检查）============
        self._execute_sells()

        # ========== 第二步：买入（情绪闸门 + 仓位约束）============
        self._execute_buys()

    def _on_date_change(self, new_date: str):
        """日期变化时的回调：重置所有账户的当日计数器"""
        # 基类单账户
        if self.portfolio:
            self.portfolio.reset_daily_counters()
        # 竞技场四账户
        if hasattr(self, 'accounts') and self.accounts:
            for pf in self.accounts.values():
                pf.reset_daily_counters()

    def _execute_sells(self, pf=None):
        """卖出执行：遍历持仓，按优先级触发卖出

        pf 传入则对该账户执行；缺省用 self.portfolio。
        """
        pf = pf if pf is not None else self.portfolio
        to_sell = []

        for code, pos in list(pf.holdings.items()):
            # 获取实时价格
            if code not in self.data_map:
                continue
            data = self.data_map[code]
            price = data.close[0]
            # pre_close = 上一根 K 线收盘价（backtrader 中索引 -1 指向前一根）
            pre_close = data.close[-1] if len(data) > 1 else price
            # 跌停价按板块计算（创业板±20% vs 主板±10%）
            limit_down = pre_close * (1.0 - self.broker_sim.limit_pct(code))

            # 更新持仓现价
            pos.mark_to_market(price)
            pos.hold_days = self._hold_days(pos.buy_date, self.current_date)

            # MA20 值，用于 B1 跌破均线判断
            ma_val = self.ma_map.get(code, {}).get(20, [None])[0] if self.ma_map else None

            # 风险检查（包含跌停判断）
            triggered = self.risk.check_all(
                pos,
                ma_value=ma_val,
                after_confirm_time=False,  # 日线回测简化
                at_close=True,  # 日线回测在收盘判断
                limit_down=limit_down,
                sealed=False,  # 简化：不判断封死
            )

            if triggered:
                to_sell.append((code, pos, triggered))

        # 执行卖出
        for code, pos, trig in to_sell:
            data = self.data_map[code]
            price = data.close[0]
            pre_close = data.close[-1] if len(data) > 1 else price
            limit_down = pre_close * (1.0 - self.broker_sim.limit_pct(code))

            fill = self.broker_sim.sell(
                code, price, pos.qty,
                pre_close=pre_close,
                limit_down=limit_down,
                ts=self.current_date,
            )

            if fill:
                pf.record_sell(fill, self.current_date, trig["reason"])

    def _execute_buys(self, pf=None, detector_name=None):
        """买入执行：情绪闸门 + 仓位约束 + 信号匹配

        pf 传入则对该账户执行；缺省用 self.portfolio。
        detector_name 传入则按该检测器过滤信号；缺省用 params.detector_name。
        """
        pf = pf if pf is not None else self.portfolio
        det_name = detector_name if detector_name is not None else self.p.detector_name
        # 情绪闸门（开关：modules.sentiment.enabled，兼容旧 ablation.sentiment.enabled；
        # 关闭时跳过闸门，按仓位上限正常开仓）
        sentiment = self.p.sentiment_scores.get(self.current_date)
        if module_enabled("sentiment.enabled"):
            gate_status = self.gate.check(sentiment)
            max_buy_round = gate_status["max_buy"]
            if gate_status["mode"] == "halted":
                return  # 暂停开仓
        else:
            max_buy_round = int(get_config("ablation.position.max_buy_per_round", 2))

        # 本轮已买入计数
        bought_this_round = 0

        # 获取当日信号
        signals_df = self.p.signals_dict.get(self.current_date)
        if signals_df is None or signals_df.empty:
            return

        # 过滤出当前策略的信号
        my_signals = signals_df[signals_df["detector"] == det_name]
        if my_signals.empty:
            return

        # 按得分排序（低分优先）
        my_signals = my_signals.sort_values("score")

        for sig in my_signals.itertuples():
            if bought_this_round >= max_buy_round:
                break

            code = sig.code
            # 检查冷却期、持仓上限等（约束开关见 modules.portfolio）
            ok, reason = pf.can_open(
                code, self.current_date, sig.score,
                round_budget=max_buy_round - bought_this_round
            )
            if not ok:
                continue

            # 获取实时数据
            if code not in self.data_map:
                continue
            data = self.data_map[code]
            price = data.close[0]
            pre_close = data.close[-1] if len(data) > 1 else price

            # 不追高 3%（开关：ablation.buy_guards.no_chase_3pct）
            if self.buy_guards["no_chase_3pct"] and sig.close > 0 and price > sig.close * 1.03:
                continue

            # 非涨停（开关：ablation.buy_guards.no_limit_up）
            # code 传入以便按板块判断（创业板涨停 ±20%）
            if self.buy_guards["no_limit_up"] and self.broker_sim.is_limit_up(price, pre_close, code):
                continue

            # 计算目标仓位
            total_value = pf.total_value({code: price})
            target_value = total_value * pf.target_weight
            qty = int(target_value / price / 100) * 100  # 整手

            if qty <= 0:
                continue

            # 下单
            fill = self.broker_sim.buy(
                code, price, qty,
                pre_close=pre_close,
                ts=self.current_date,
                available_cash=pf.broker_cash,
            )

            if fill:
                pf.record_buy(fill, self.current_date, sig.score)
                pf.buy_count_today += 1
                bought_this_round += 1

    def _hold_days(self, buy_date: str, current_date: str) -> int:
        from datetime import date as _d
        da = _d.fromisoformat(buy_date[:10])
        db = _d.fromisoformat(current_date[:10])
        return (db - da).days

    def _force_close(self, pf=None, reason="回测结束强制卖出"):
        """回测最后一天：按当日收盘价强制卖出全部剩余持仓

        撮合层在跌停封死时拒绝成交，但收口清仓是统一结算动作，
        broker 拒单时直接构造 Fill（费用照常计）强制平仓，
        保证每个账户最后都不留持仓。
        清仓后立即用现金口径重记当日净值，最终净值即现金口径。
        """
        pf = pf if pf is not None else self.portfolio
        if not pf.holdings:
            return
        date = self.current_date or (pf.daily_nav[-1]["date"] if pf.daily_nav else "")
        for code, pos in list(pf.holdings.items()):
            if code in self.data_map:
                data = self.data_map[code]
                price = data.close[0]
                pre_close = data.close[-1] if len(data) > 1 else price
            else:  # 无行情数据的持仓：按最新成本价估算清算价
                price = pos.current_price or pos.init_price
                pre_close = price
            limit_down = pre_close * (1.0 - self.broker_sim.limit_pct(code))
            fill = self.broker_sim.sell(
                code, price, pos.qty,
                pre_close=pre_close,
                limit_down=limit_down,
                ts=date,
            )
            if fill is None:
                fees = self.broker_sim.calc_fees(price, pos.qty, "sell")
                fill = Fill(code, "sell", price, pos.qty, fees["commission"],
                            fees["stamp_tax"], fees["transfer_fee"],
                            fees["total"], date)
            pf.record_sell(fill, date, reason)
        pf.mark_nav(date, {})

    def stop(self):
        """策略结束：最后一天强制清仓 + 输出统计"""
        self._force_close(self.portfolio)
        total_trades = len(self.portfolio.trades)
        buys = len([t for t in self.portfolio.trades if t["side"] == "buy"])
        sells = len([t for t in self.portfolio.trades if t["side"] == "sell"])
        final_nav = self.portfolio.daily_nav[-1]["nav"] if self.portfolio.daily_nav else 1.0

        print(f"[{self.p.account_id}] 策略结束:")
        print(f"  最终净值: {final_nav:.4f} ({(final_nav-1)*100:.2f}%)")
        print(f"  总交易: {total_trades} 笔 (买{buys} 卖{sells})")
        print(f"  持仓数: {len(self.portfolio.holdings)}")
