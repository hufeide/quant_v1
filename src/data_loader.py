"""日线数据加载与缓存"""
import pandas as pd
from pathlib import Path
from functools import lru_cache


def load_daily(code: str, daily_dir: str = "data/daily") -> pd.DataFrame:
    """加载单只股票日线并附加指标"""
    path = Path(daily_dir) / f"{code}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"日线不存在: {path}")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def load_daily_with_indicators(code: str, daily_dir: str = "data/daily") -> pd.DataFrame:
    """加载并计算指标"""
    from src.indicators import add_indicators
    df = load_daily(code, daily_dir)
    # 以 atr14 为准判断指标是否完整（新增指标后老缓存只缺 atr14）
    if "atr14" not in df.columns:
        df = add_indicators(df)
    return df


def list_codes(daily_dir: str = "data/daily") -> list:
    """列出可用股票代码"""
    p = Path(daily_dir)
    if not p.exists():
        return []
    return sorted(f.stem for f in p.glob("*.parquet"))
