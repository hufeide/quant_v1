#!/usr/bin/env python3
"""竞技场回测主入口

流程：
  1. 加载日线数据（data/daily/*.parquet）
  2. 运行信号引擎，生成 data/signals/all_signals.parquet
  3. 加载情绪指数
  4. 构建 Cerebro，注入信号与情绪
  5. 运行单策略或四策略竞技场
  6. 汇总结果，输出净值曲线与交易明细

用法：
  python backtrader/run_arena.py --strategies pullback momentum
  python backtrader/run_arena.py --all              # 四策略竞技场
  python backtrader/run_arena.py --single oversold
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, get_config
from src.data_loader import list_codes
from src.signal_engine import generate_signals
from bt_engine.cerebro import load_sentiment, load_signals, run_arena


def plot_equity_curves(results: dict, out_dir: str):
    """绘制四策略净值曲线对比图"""
    plt.figure(figsize=(12, 6))
    for name, strat in results.items():
        nav = [r["nav"] for r in strat.portfolio.daily_nav]
        dates = [r["date"] for r in strat.portfolio.daily_nav]
        if nav:
            plt.plot(dates, nav, label=f"{name} ({(nav[-1]-1)*100:+.1f}%)", linewidth=1.5)

    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.4, label="初始净值")
    plt.title("策略竞技场 - 四账户净值曲线")
    plt.xlabel("日期")
    plt.ylabel("净值")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = Path(out_dir) / "equity_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[图] 净值曲线已保存: {out}")


def collect_trades(results: dict) -> pd.DataFrame:
    """汇总所有账户交易明细"""
    all_trades = []
    for name, strat in results.items():
        for t in strat.portfolio.trades:
            t["strategy"] = name
            all_trades.append(t)
    df = pd.DataFrame(all_trades)
    return df


def collect_daily_nav(results: dict) -> pd.DataFrame:
    """汇总所有账户净值序列"""
    all_nav = []
    for name, strat in results.items():
        for r in strat.portfolio.daily_nav:
            r["strategy"] = name
            all_nav.append(r)
    return pd.DataFrame(all_nav)


def run_single(strategy_name: str, codes, signals, sentiment, args, out_dir):
    """运行单策略回测"""
    from bt_engine.cerebro import run_backtest
    strat = run_backtest(
        codes=codes, signals=signals, sentiment=sentiment,
        strategy_name=strategy_name,
        account_id=strategy_name.upper(),
        initial_cash=args.cash,
    )
    return {strategy_name: strat}


def main():
    ap = argparse.ArgumentParser(description="V1 策略竞技场回测")
    ap.add_argument("--all", action="store_true", help="运行全部四策略")
    ap.add_argument("--single", type=str, help="运行单策略 (pullback/breakout/oversold/momentum)")
    ap.add_argument("--strategies", nargs="+", default=["pullback","breakout","oversold","momentum"])
    ap.add_argument("--cash", type=float, default=1_000_000, help="每账户初始资金")
    ap.add_argument("--no-signals", action="store_true", help="跳过信号生成（使用已有）")
    ap.add_argument("--codes", nargs="+", help="指定股票代码列表（测试用）")
    ap.add_argument("--out", default="backtest/results", help="输出目录")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载股票代码
    if args.codes:
        codes = args.codes
    else:
        codes = list_codes(get_config("paths.daily", "data/daily"))
    if not codes:
        print("[错误] 无可用日线数据，请先运行数据生成或下载脚本")
        return

    print(f"[回测] 股票池: {len(codes)} 只")

    # 生成信号
    if not args.no_signals:
        # 获取所有交易日
        from src.data_loader import load_daily
        sample = load_daily(codes[0])
        dates = [str(d)[:10] for d in sample["date"]
                 if get_config("project.backtest_start", "2025-06-12") <= str(d)[:10]
                 <= get_config("project.backtest_end", "2025-11-02")]
        print(f"[回测] 信号生成: {len(dates)} 个交易日")
        signals_df = generate_signals(run_dates=dates, codes=codes)
        signals = {str(d): g for d, g in signals_df.groupby("date")} if not signals_df.empty else {}
    else:
        signals = load_signals(get_config("paths.signals", "data/signals"))

    # 加载情绪
    sentiment = load_sentiment(get_config("paths.sentiment", "data/sentiment"))
    print(f"[回测] 情绪指数: {len(sentiment)} 天")

    # 运行回测
    if args.single:
        results = run_single(args.single, codes, signals, sentiment, args, out_dir)
    else:
        results = run_arena(
            codes=codes, signals=signals, sentiment=sentiment,
            strategies=args.strategies, initial_cash=args.cash,
        )

    # 汇总输出
    nav_df = collect_daily_nav(results)
    nav_df.to_parquet(out_dir / "daily_nav.parquet", index=False)

    trades_df = collect_trades(results)
    trades_df.to_parquet(out_dir / "trades.parquet", index=False)

    # 净值曲线图
    if len(results) > 1:
        plot_equity_curves(results, str(out_dir))

    # 各策略汇总
    summary = {}
    for name, strat in results.items():
        nav_list = strat.portfolio.daily_nav
        if not nav_list:
            summary[name] = {"error": "无净值记录"}
            continue
        final_nav = nav_list[-1]["nav"]
        total_ret = (final_nav - 1) * 100
        trades = strat.portfolio.trades
        sells = [t for t in trades if t["side"] == "sell"]
        wins = [t for t in sells if t.get("gain_pct", 0) > 0]
        win_rate = len(wins) / len(sells) * 100 if sells else 0.0

        summary[name] = {
            "final_nav": round(final_nav, 4),
            "total_return_pct": round(total_ret, 2),
            "total_trades": len(trades),
            "buys": len([t for t in trades if t["side"] == "buy"]),
            "sells": len(sells),
            "win_rate_pct": round(win_rate, 1),
            "final_holdings": len(strat.portfolio.holdings),
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("回测结果汇总")
    print("="*50)
    for name, s in summary.items():
        print(f"  {name:12s}  收益率 {s.get('total_return_pct',0):+6.2f}%  "
              f"交易 {s.get('total_trades',0):3d}笔  胜率 {s.get('win_rate_pct',0):.1f}%")


if __name__ == "__main__":
    main()
