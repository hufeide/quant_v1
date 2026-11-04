"""最小数据生成器：为快速跑通流程生成模拟日线数据

生成 N 只股票 × M 个交易日的 OHLCV 数据，
支持注入特定形态（缩量回踩、平台突破、超跌反弹、强者恒强）用于检测器测试。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def generate_stock_data(
    code: str,
    days: int = 300,
    start_price: float = 20.0,
    seed: int = None,
    inject_pattern: str = None,  # "pullback", "breakout", "oversold", "momentum"
    inject_date: int = None,     # 注入形态的日期（交易日索引）
) -> pd.DataFrame:
    """生成单只股票的日线数据"""
    rng = np.random.default_rng(seed if seed is not None else hash(code) % 100000)

    # 生成日期
    dates = pd.bdate_range(end=datetime(2025, 11, 2), periods=days)

    # 随机游走基础价格
    drift = rng.normal(0.0002, 0.001)
    prices = [start_price]
    for i in range(days - 1):
        ret = rng.normal(drift, 0.02)
        prices.append(prices[-1] * (1 + ret))

    closes = np.array(prices)
    opens = closes * (1 + rng.normal(0, 0.005, days))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.008, days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.008, days)))
    # 用 float64 避免 pandas 3.0 对 int 列赋浮点的 LossySetitemError
    volumes = rng.integers(1000000, 10000000, days).astype(np.float64)
    amounts = volumes * closes

    df = pd.DataFrame({
        'code': code,
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'amount': amounts,
        'pct_chg': (closes[1:] / closes[:-1] - 1).tolist() + [0.0],
        'turnover_rate': rng.uniform(0.5, 5.0, days),
        'market_cap_float': rng.uniform(5e9, 5e10),
        'is_st': False,
    })

    # 注入特定形态
    if inject_pattern and inject_date is not None and inject_date < days - 20:
        df = _inject_pattern(df, inject_pattern, inject_date, rng)

    # 注入改动了 close，需重算 pct_chg
    if inject_pattern:
        df['pre_close'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['pct_chg'] = (df['close'] / df['pre_close'] - 1.0) * 100.0

    # 计算衍生指标
    df['pre_close'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['limit_up'] = df['pre_close'] * 1.10
    df['limit_down'] = df['pre_close'] * 0.90

    # 创业板（300/301开头）涨跌幅±20%
    if code.startswith('30'):
        df['limit_up'] = df['pre_close'] * 1.20
        df['limit_down'] = df['pre_close'] * 0.80

    return df


def _inject_pattern(df: pd.DataFrame, pattern: str, idx: int, rng) -> pd.DataFrame:
    """在指定位置注入形态，方便检测器测试"""
    df = df.copy()
    if pattern == "pullback":
        # 上涨8% → 缩量回调
        for i in range(idx - 10, idx):
            df.loc[i, 'close'] *= 1.008
            df.loc[i, 'volume'] *= 1.5
        for i in range(idx, idx + 5):
            df.loc[i, 'close'] *= 0.97
            df.loc[i, 'volume'] *= 0.5
    elif pattern == "breakout":
        # 前60日新高 + 放量
        base = df.loc[idx-1, 'close']
        for i in range(idx-60, idx):
            df.loc[i, 'close'] = base * 0.92
        for i in range(idx, idx+2):
            df.loc[i, 'close'] = base * 1.05
            df.loc[i, 'volume'] *= 2.0
    elif pattern == "oversold":
        # 10日跌18% + 当日阳线（留出余量，确保越过 15% 阈值）
        for i in range(idx-10, idx):
            df.loc[i, 'close'] *= 0.982
        df.loc[idx, 'close'] = df.loc[idx-1, 'close'] * 1.03
        # 当日最低不破前10日最低，满足"不创新低"
        prev_min_low = df.loc[idx-10:idx-1, 'low'].min()
        df.loc[idx, 'low'] = max(df.loc[idx, 'open'], prev_min_low) * 1.001
        df.loc[idx, 'pct_chg'] = (df.loc[idx, 'close'] / df.loc[idx-1, 'close'] - 1) * 100
    elif pattern == "momentum":
        # 20日涨25% + 守住MA10
        base = df.loc[idx-20, 'close']
        for i in range(idx-20, idx+1):
            df.loc[i, 'close'] = base * 1.25 * (1 + 0.01 * (i - idx))
    return df


def generate_universe(n_stocks: int = 50, days: int = 300, output_dir: str = "data/daily"):
    """生成最小股票池的模拟数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 混合各板块代码
    codes = []
    for i in range(n_stocks):
        if i < 20:
            codes.append(f"000{i:03d}")       # 主板
        elif i < 35:
            codes.append(f"002{i-20:03d}")    # 中小板
        else:
            codes.append(f"300{i-35:03d}")    # 创业板

    patterns = ["pullback", "breakout", "oversold", "momentum", None]
    injected = []

    for code in codes:
        # 注入位置至少 60，确保有足够的预热数据
        idx = 60 + hash(code) % (days - 60)
        pattern = patterns[hash(code) % len(patterns)]
        df = generate_stock_data(code, days=days, seed=hash(code) % 10000,
                                 inject_pattern=pattern, inject_date=idx)
        df.to_parquet(output_path / f"{code}.parquet", index=False)
        if pattern:
            injected.append({"code": code, "pattern": pattern, "date_idx": idx})

    print(f"✓ 生成 {len(codes)} 只股票 × {days} 个交易日")
    print(f"  形态注入: {injected}")
    return injected


def generate_snapshot(codes: list, date: str = "2025-09-15", output_dir: str = "data/snapshot"):
    """生成某日全市场快照"""
    rng = np.random.default_rng(hash(date) % 10000)
    rows = []
    for code in codes:
        cap = float(code[2:])
        close = rng.uniform(5, 100)
        if code.startswith("30"):
            limit = 20.0
        else:
            limit = 10.0
        rows.append({
            "date": date,
            "code": code,
            "close": round(close, 2),
            "pct_chg": round(rng.uniform(-limit, limit), 2),
            "amount": round(rng.uniform(5e7, 5e9), 0),
            "turnover_rate": round(rng.uniform(0.5, 15.0), 2),
            "market_cap_float": round(rng.uniform(3e9, 2e12), 0),
            "is_st": False,
            "pre_close": round(close / (1 + rng.uniform(-0.03, 0.03)), 2),
        })
    df = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{date}.parquet", index=False)
    print(f"✓ 快照 {date}: {len(df)} 只")
    return df


def generate_sentiment(dates: list, output_dir: str = "data/sentiment"):
    """生成情绪指数"""
    rng = np.random.default_rng(42)
    rows = []
    for d in dates:
        rows.append({
            "date": d,
            "profit_effect": round(rng.uniform(30, 80), 1),
            "market_temp": round(rng.uniform(25, 85), 1),
            "retail_sentiment": round(rng.uniform(20, 90), 1),
        })
    df = pd.DataFrame(rows)
    # 计算综合分
    df["sentiment_score"] = (
        df["profit_effect"] * 0.45 +
        df["market_temp"] * 0.35 +
        df["retail_sentiment"] * 0.20
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "sentiment.parquet", index=False)
    print(f"✓ 情绪指数: {len(df)} 天")
    return df


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    injected = generate_universe(n_stocks=n, days=days)

    # 生成几个快照日期
    for d in ["2025-09-10", "2025-09-15", "2025-09-20"]:
        generate_snapshot(codes=[f"000{i:03d}" for i in range(n)], date=d)

    # 生成情绪指数
    generate_sentiment(dates=["2025-09-10", "2025-09-11", "2025-09-12", "2025-09-15"])
