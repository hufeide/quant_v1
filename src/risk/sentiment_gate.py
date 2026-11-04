"""情绪闸门：只限买，不限卖

综合分 = 0.45*赚钱效应 + 0.35*市场温度 + 0.20*散户情绪

三档：
  > gate_open  -> 正常开仓
  > gate_limit -> 每轮最多开 1 只
  <= gate_limit -> 暂停一切新开仓

关键设计：
  · 闸门只管买，止损永远照常执行
  · 情绪数据缺失时 fail-open 放行
"""
from typing import Dict, Any, Optional


class SentimentGate:
    """情绪闸门"""

    OPEN = "open"        # 正常开仓
    LIMITED = "limited"  # 每轮最多 1 只
    HALTED = "halted"    # 暂停开仓
    UNKNOWN = "unknown"  # 数据缺失（按 fail_open 处理）

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def combine(self, profit_effect: float, market_temp: float,
                retail_sentiment: float) -> float:
        """合成综合情绪分"""
        w = self.cfg.get("weights", {})
        score = (
            profit_effect * w.get("profit_effect", 0.45) +
            market_temp * w.get("market_temp", 0.35) +
            retail_sentiment * w.get("retail_sentiment", 0.20)
        )
        return round(score, 2)

    def classify(self, score: float) -> str:
        """按阈值分档"""
        open_thr = self.cfg.get("gate_open", 52.0)
        limit_thr = self.cfg.get("gate_limit", 40.0)

        if score > open_thr:
            return self.OPEN
        elif score > limit_thr:
            return self.LIMITED
        else:
            return self.HALTED

    def max_buy_per_round(self, score: Optional[float]) -> int:
        """返回本轮允许的最大开仓数量"""
        if score is None:
            if self.cfg.get("fail_open", True):
                return int(self.cfg.get("max_buy_per_round_default", 2))
            return 0

        mode = self.classify(score)
        if mode == self.OPEN:
            return int(self.cfg.get("max_buy_per_round_default", 2))
        elif mode == self.LIMITED:
            return 1
        elif mode == self.HALTED:
            return 0
        else:
            return int(self.cfg.get("max_buy_per_round_default", 2))

    def allows_buy(self, score: Optional[float]) -> bool:
        """是否允许开新仓"""
        return self.max_buy_per_round(score) > 0

    def check(self, score: Optional[float]) -> Dict[str, Any]:
        """完整检查，返回状态与说明"""
        if score is None:
            return {"mode": self.UNKNOWN,
                    "action": "放行" if self.cfg.get("fail_open", True) else "阻止",
                    "max_buy": int(self.cfg.get("max_buy_per_round_default", 2)) if self.cfg.get("fail_open", True) else 0,
                    "note": "情绪数据缺失，按 fail_open 处理"}

        mode = self.classify(score)
        return {
            "mode": mode,
            "score": score,
            "max_buy": self.max_buy_per_round(score),
            "action": {
                self.OPEN: "正常开仓",
                self.LIMITED: "每轮限开 1 只",
                self.HALTED: "暂停新开仓",
            }[mode],
        }
