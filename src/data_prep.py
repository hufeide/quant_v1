"""从分钟数据聚合日线，以及下载真实日线（akshare）的接口

若本地已有日线 parquet（data/daily/*.parquet），可直接用，
本模块提供从分钟线聚合与从 akshare 拉取两条路径。
"""
from pathlib import Path
import numpy as np
import pandas as pd

DAILY_COLUMNS = [
    "code", "date", "open", "high", "low", "close",
    "volume", "amount", "pct_chg", "turnover_rate",
    "market_cap_float", "is_st", "pre_close",
    "limit_up", "limit_down",
]


def aggregate_minute_to_daily(minute_df: pd.DataFrame) -> pd.DataFrame:
    """把分钟线聚合为日线

    minute_df 需含列：code/datetime/open/high/low/close/volume/amount
    """
    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")

    agg = df.groupby(["code", "date"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    ).reset_index()

    agg["pre_close"] = agg.groupby("code")["close"].shift(1)
    agg["pre_close"] = agg["pre_close"].fillna(agg.groupby("code")["open"].transform("first"))

    agg["pct_chg"] = (agg["close"] / agg["pre_close"] - 1.0) * 100.0
    agg["limit_up"] = agg["pre_close"] * 1.10
    agg["limit_down"] = agg["pre_close"] * 0.90

    # 创业板 ±20%
    pc_mask = agg["code"].astype(str).str.startswith("30")
    agg.loc[pc_mask, "limit_up"] = agg.loc[pc_mask, "pre_close"] * 1.20
    agg.loc[pc_mask, "limit_down"] = agg.loc[pc_mask, "pre_close"] * 0.80

    return agg


def save_daily(daily_df: pd.DataFrame, daily_dir: str = "data/daily"):
    """按 code 拆分并保存日线到 parquet"""
    out = Path(daily_dir)
    out.mkdir(parents=True, exist_ok=True)

    for code, g in daily_df.groupby("code"):
        g = g.sort_values("date").reset_index(drop=True)
        g.to_parquet(out / f"{code}.parquet", index=False)

    return len(daily_df.groupby("code"))


def fetch_akshare_daily(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从 akshare 拉取单只股票前复权日线

    未安装 akshare 时抛出 ImportError，由调用方降级到模拟数据。
    """
    import akshare as ak

    symbol = code.zfill(6)
    raw = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""), adjust="qfq",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    rename = {
        "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
        "换手率": "turnover_rate", "流通市值": "market_cap_float",
    }
    df = raw.rename(columns=rename)
    df["code"] = symbol
    df["is_st"] = df.get("名称", "").astype(str).str.contains("ST", na=False) \
        if "名称" in df.columns else False

    keep = [c for c in DAILY_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["pre_close"] = df["close"].shift(1)
    df["pct_chg"] = df["close"].pct_change() * 100.0
    df["limit_up"] = df["pre_close"] * 1.10
    df["limit_down"] = df["pre_close"] * 0.90

    pc = df["code"].astype(str).str.startswith("30")
    df.loc[pc, "limit_up"] = df.loc[pc, "pre_close"] * 1.20
    df.loc[pc, "limit_down"] = df.loc[pc, "pre_close"] * 0.80

    return df
