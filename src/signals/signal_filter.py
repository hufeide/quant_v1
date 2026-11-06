"""信号过滤器：原始信号 -> 有效信号（不直接下单）

职责边界：
    只回答"哪些原始信号真正有效"。不决定买多少股，不下单，
    不修改持仓状态，不修改信号引擎内部状态。

过滤规则（全部由 config.yaml 的 modules.detectors / modules.scoring 开关控制）：
    1. 检测器启用开关（modules.detectors.<name>，兼容 legacy detectors.<name>.enabled）
    2. 打分阈值（modules.scoring.enabled 时生效：score >= buy_max_score 视为过热，不买入）
    3. 点在时间候选池（universe）成员资格（防未来函数，data.xlsx 点在时间快照）
    4. 价格与 ATR 有效性（entry_price > 0，ATR 有限且 >= 0）

输入：signal_engine 产出的原始信号 DataFrame。
输出：通过全部过滤的有效信号 DataFrame（列结构不变）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd

from src.config import get_config, load_config, module_enabled


def detector_enabled(name: str) -> bool:
    """检测器级开关：modules.detectors.<name> 优先，兼容 legacy detectors.<name>.enabled"""
    cfg = load_config()
    legacy = (cfg.get("detectors", {}).get(name, {}).get("enabled", True)
              if isinstance(cfg.get("detectors", {}).get(name), dict) else True)
    return module_enabled(f"detectors.{name}", default=True) and legacy


def _universe_membership(row, universe_by_date: Dict[str, Optional[List[str]]]) -> bool:
    """点在时间 universe 成员资格：universe_by_date[date] 为 None 时表示不启用"""
    date_str = str(row["date"])[:10]
    codes = universe_by_date.get(date_str)
    if codes is None:
        return True
    return row["code"] in codes


def filter_signals(
    raw: pd.DataFrame,
    *,
    universe_by_date: Optional[Dict[str, Optional[List[str]]]] = None,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """原始信号 -> 有效信号（纯函数，不修改输入 df）"""
    if raw is None or isinstance(raw, pd.Series):
        if isinstance(raw, pd.Series):
            raw = raw.to_frame().T
        else:
            return pd.DataFrame()
    if raw.empty:
        return raw
    cfg = cfg if cfg is not None else load_config()

    keep = pd.Series(True, index=raw.index)

    # 1. 检测器启用开关
    if cfg.get("_detector_filter", True):
        enabled = {name: detector_enabled(name) for name in raw.get("detector", pd.Series(dtype=str)).unique()}
        keep &= raw["detector"].map(lambda d: enabled.get(d, True))

    # 2. 打分阈值（scoring.enabled 时生效："score >= buy_max_score" 视为过热，不买入）
    if module_enabled("scoring.enabled", default=False):
        buy_max = float(cfg.get("scoring", {}).get("buy_max_score", 75.0))
        keep &= raw["score"] < buy_max

    # 3. 点在时间 universe 成员资格
    if universe_by_date:
        keep &= raw.apply(_universe_membership, args=(universe_by_date,), axis=1)

    # 4. 价格 / ATR 有效性
    keep &= (pd.to_numeric(raw.get("entry_price", 0), errors="coerce").fillna(0) > 0)
    atr = pd.to_numeric(raw.get("atr", 0), errors="coerce")
    keep &= atr.apply(lambda v: not math.isnan(v) and v >= 0) if "atr" in raw.columns else True

    return raw[keep].reset_index(drop=True)


def filter_detector(raw: pd.DataFrame, detector: str) -> pd.DataFrame:
    """按检测器名过滤（策略层消费信号时使用）"""
    if raw is None or raw.empty:
        return pd.DataFrame()
    return raw[raw["detector"] == detector].reset_index(drop=True)
