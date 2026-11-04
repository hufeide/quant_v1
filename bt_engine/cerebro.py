"""Cerebro 组装：构建竞技场回测环境

六账户设计说明：
  backtrader 的一个 Cerebro 通常配一个 broker + 一个策略。
  V1 竞技场有 4 个自动账户，每个账户独立的 Portfolio（现金/持仓/冷却期）。
  因此本方案：一个 Cerebro + 一个策略实例，策略内部用 ArenaArena
  （portfolio_arena）管理多账户，每个账户独立记账。

本文件负责：
  - 组装数据 feed
  - 注入信号与情绪到策略 params
  - 跑 Cerebro
  - 回收各账户净值/成交明细
"""
import backtrader as bt
import pandas as pd
from typing import Dict, List
from src.config import load_config, get_config
from src.data_loader import list_codes
from bt_engine.datafeed import StockDataFeed, add_data_feed
from strategies.arena_strategies import get_strategy_class


def load_sentiment(sentiment_dir: str = "data/sentiment") -> Dict[str, float]:
    """读取情绪指数 -> {date: score}"""
    from pathlib import Path
    p = Path(sentiment_dir) / "sentiment.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if "date" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        d = str(row["date"])[:10]
        out[d] = float(row["sentiment_score"])
    return out


def load_signals(signals_dir: str = "data/signals") -> Dict[str, "object"]:
    """读取信号表 -> {date: DataFrame}"""
    from pathlib import Path
    import pandas as pd
    p = Path(signals_dir) / "all_signals.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if "date" not in df.columns:
        return {}
    return {d: g for d, g in df.groupby("date")}


def build_cerebro(
    codes: List[str],
    fromdate=None,
    todate=None,
    signals: Dict = None,
    sentiment: Dict = None,
    initial_cash: float = 1_000_000,
    strategy_name: str = "pullback",
    account_id: str = "A1",
):
    """构建 Cerebro 实例"""
    cerebro = bt.Cerebro(stdstats=False)

    daily_dir = get_config("paths.daily", "data/daily")
    for code in codes:
        d = add_data_feed(cerebro, code, daily_dir, fromdate, todate)
        if d is None:
            print(f"[警告] {code} 数据为空，跳过")

    strat_cls = get_strategy_class(strategy_name)

    cerebro.addstrategy(
        strat_cls,
        signals_dict=signals or {},
        sentiment_scores=sentiment or {},
        initial_cash=initial_cash,
        account_id=account_id,
        detector_name=strategy_name,
    )

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.set_coc(True)  # 当日收盘成交

    return cerebro


def run_backtest(
    codes: List[str],
    signals: Dict,
    sentiment: Dict,
    strategy_name: str = "pullback",
    account_id: str = "A1",
    initial_cash: float = 1_000_000,
    fromdate=None,
    todate=None,
):
    """运行单策略回测，返回策略实例"""
    cerebro = build_cerebro(
        codes=codes,
        fromdate=fromdate,
        todate=todate,
        signals=signals,
        sentiment=sentiment,
        initial_cash=initial_cash,
        strategy_name=strategy_name,
        account_id=account_id,
    )

    print(f"[回测] 启动 {account_id} ({strategy_name}) 资金 {initial_cash:,.0f}")
    results = cerebro.run()
    strat = results[0]
    return strat


def run_arena(
    codes: List[str],
    signals: Dict,
    sentiment: Dict,
    strategies: List[str],
    initial_cash: float = 1_000_000,
    fromdate=None,
    todate=None,
):
    """依次运行四策略，返回 {account_id: strategy}"""
    out = {}
    account_map = {"pullback": "A1", "breakout": "A2",
                   "oversold": "A3", "momentum": "A4"}
    for name in strategies:
        s = run_backtest(
            codes=codes,
            signals=signals,
            sentiment=sentiment,
            strategy_name=name,
            account_id=account_map.get(name, name),
            initial_cash=initial_cash,
            fromdate=fromdate,
            todate=todate,
        )
        out[name] = s
    return out
