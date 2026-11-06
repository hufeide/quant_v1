"""成交与交易记录的统计层（不直接下单、不修改持仓账簿）

职责边界：
    只负责"对已成交的交易做记录/统计"。成交日志（trades 列表）
    仍由 Portfolio 统一维护（record_buy / record_sell 写入现金、
    持仓、净值、交易明细），本模块只读取并产出统计视图，避免重复记账。

架构位置：execution 层（与 order_manager 平级），是回测/Web 汇总的
分析报告器，不参与交易决策，也不持有任何持仓状态。
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd


class TradeRecorder:
    """交易统计器：基于 Portfolio.trades 产出汇总"""

    def __init__(self, portfolio=None):
        self.portfolio = portfolio
        # 直接引用 Portfolio 的 trades（同一份列表），不复制、不另记
        self.trades = portfolio.trades if portfolio is not None else []

    # ---------- 统计 ----------

    def summary(self) -> dict:
        """单账户交易统计（供回测汇总/Web 展示）"""
        trades = self.trades
        sells = [t for t in trades if t["side"] == "sell"]
        buys = [t for t in trades if t["side"] == "buy"]
        wins = [t for t in sells if (t.get("gain_pct", 0) or 0) > 0]
        total_gain = sum(t.get("gain_pct", 0) or 0 for t in sells)
        avg_hold = (sum(t.get("hold_days", 0) or 0 for t in sells) / len(sells)) if sells else 0.0
        return {
            "account": self.portfolio.account_id if self.portfolio else "",
            "total_trades": len(trades),
            "buys": len(buys),
            "sells": len(sells),
            "win_rate": len(wins) / len(sells) if sells else 0.0,
            "total_gain_pct": round(total_gain, 2),
            "avg_hold_days": round(avg_hold, 1),
        }

    def win_rate(self) -> float:
        sells = [t for t in self.trades if t["side"] == "sell"]
        if not sells:
            return 0.0
        wins = [t for t in sells if (t.get("gain_pct", 0) or 0) > 0]
        return len(wins) / len(sells)

    def to_frame(self) -> pd.DataFrame:
        """交易明细转 DataFrame"""
        return pd.DataFrame(self.trades)

    def recorded_trades(self) -> List[dict]:
        """原始成交明细（引用，不复制）"""
        return self.trades


def aggregate_trades(portfolio_summaries: List[dict]) -> dict:
    """把多账户的 summary 合并成竞技场总览"""
    total_trades = sum(s["total_trades"] for s in portfolio_summaries)
    buys = sum(s["buys"] for s in portfolio_summaries)
    sells = sum(s["sells"] for s in portfolio_summaries)
    total_gain = sum(s.get("total_gain_pct", 0) or 0 for s in portfolio_summaries)
    wins_total = 0.0
    sells_total = 0
    for s in portfolio_summaries:
        wins_total += s.get("win_rate", 0.0) * (s.get("sells", 0) or 1)
        sells_total += s.get("sells", 0)
    return {
        "total_trades": total_trades,
        "buys": buys,
        "sells": sells,
        "total_gain_pct": round(total_gain, 2),
        "accounts": len(portfolio_summaries),
        "agg_win_rate": round(wins_total / sells_total, 4) if sells_total else 0.0,
    }
