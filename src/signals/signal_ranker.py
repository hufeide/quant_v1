"""信号排序器：有效信号 -> 排序结果（不直接下单）

职责边界：
    只回答"在有效信号里按什么顺序考虑"。不决定买多少股，不下单，
    不修改持仓状态，不修改原始信号本身的评分。

排序规则（当前策略语义：低分优先——过热检测器打分越低越先买；
可按 config.yaml 的 ablation.order.by 切换）：
    by=score  按综合评分升序（默认，与 arena 现有行为一致）
    by=atr    按 ATR 升序（先买波动小的，风险预算更省）
    by=amount 按成交额升序（先买流动性好的）

输入：signal_filter 产出的有效信号 DataFrame。
输出：排序后的 DataFrame（列结构不变，新增可选的 rank 列）。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def rank_signals(
    valid: pd.DataFrame,
    *,
    order_by: str = "score",
    ascending: bool = True,
    add_rank: bool = True,
) -> pd.DataFrame:
    """有效信号 -> 排序结果（纯函数，不修改输入 df）"""
    if valid is None or valid.empty:
        return valid if valid is not None else pd.DataFrame()

    out = valid.copy()
    if order_by not in out.columns:
        order_by = "score"

    # 复合键：主键 + code 兜底，保证排序稳定可复现
    out = out.sort_values([order_by, "code"], ascending=[ascending, True])

    if add_rank:
        out = out.reset_index(drop=True)
        out["rank"] = list(range(1, len(out) + 1))
    return out
