"""候选池粗筛：从全市场快照筛出候选股。

每条过滤条件都可独立开关（universe.filters.*），
这样消融回测可以单独关掉某一条，测量它的贡献。
filter_universe 返回 (df, funnel)，funnel 记录每层筛掉了多少。
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from src.config import get_config


def _on(name: str, default: bool = True) -> bool:
    """读取 universe.filters.<name> 开关"""
    return bool(get_config(f"universe.filters.{name}", default))


def filter_universe(snapshot: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """对全市场快照应用硬过滤条件

    Returns:
        (筛选后的 DataFrame, funnel 统计 dict)
        funnel: {"raw": N, "st": x, "price": y, ...} 每层剩余数量
    """
    df = snapshot.copy()
    funnel = {"raw": int(len(df))}
    n_before = int(len(df))

    # 1. 剔除 ST
    if _on("exclude_st"):
        df = df[~df["is_st"]]
        funnel["st_excluded"] = n_before - int(len(df))
        n_before = int(len(df))
    else:
        funnel["st_excluded"] = 0

    # 2. 股价区间
    if _on("price_range"):
        before = int(len(df))
        df = df[(df["close"] >= get_config("universe.min_price")) &
                (df["close"] <= get_config("universe.max_price"))]
        funnel["price_excluded"] = before - int(len(df))
    else:
        funnel["price_excluded"] = 0

    # 3. 流通市值区间
    if _on("market_cap_range"):
        before = int(len(df))
        df = df[(df["market_cap_float"] >= get_config("universe.min_float_mv")) &
                (df["market_cap_float"] <= get_config("universe.max_float_mv"))]
        funnel["market_cap_excluded"] = before - int(len(df))
    else:
        funnel["market_cap_excluded"] = 0

    # 4. 涨跌幅限制（不追高不接跌）
    if _on("pct_chg_limit"):
        before = int(len(df))
        df = df[(df["pct_chg"] <= get_config("universe.max_daily_pct_up")) &
                (df["pct_chg"] >= get_config("universe.max_daily_pct_down"))]
        funnel["pct_chg_excluded"] = before - int(len(df))
    else:
        funnel["pct_chg_excluded"] = 0

    # 5. 流动性：按成交额排序取前 N
    if _on("amount_top_n"):
        pool_size = int(get_config("universe.candidate_pool_size", 600))
        before = int(len(df))
        df = df.sort_values("amount", ascending=False).head(pool_size)
        funnel["amount_excluded"] = max(0, before - int(len(df)))
        funnel["candidate_pool_size"] = int(len(df))
    else:
        funnel["amount_excluded"] = 0
        funnel["candidate_pool_size"] = int(len(df))

    # 6. 换手率过热排除（9 月新增）
    if _on("overheat_exclude") and get_config("universe.overheat_exclude_enabled", False):
        before = int(len(df))
        thr = df["turnover_rate"].quantile(
            get_config("universe.overheat_top_pct", 20) / 100.0)
        overheat_mask = (df["turnover_rate"] >= thr) & \
                        (df["market_cap_float"] < get_config("universe.overheat_min_float_mv"))
        funnel["overheat_excluded"] = int(overheat_mask.sum())
        df = df[~overheat_mask]
    else:
        funnel["overheat_excluded"] = 0

    funnel["final"] = int(len(df))
    return df.reset_index(drop=True), funnel


def load_snapshot(date: str, snapshot_dir: str = None) -> pd.DataFrame:
    """加载某日快照"""
    if snapshot_dir is None:
        snapshot_dir = get_config("paths.snapshot", "data/snapshot")
    path = Path(snapshot_dir) / f"{date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"快照不存在: {path}")
    return pd.read_parquet(path)


def build_candidate_pool(date: str) -> Tuple[pd.DataFrame, Dict]:
    """完整候选池构建流程，返回 (池, funnel)"""
    snap = load_snapshot(date)
    return filter_universe(snap)
