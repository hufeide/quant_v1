"""超跌反弹检测器：均值回归 / 抄底"""
from typing import Dict, Any
import pandas as pd
from src.signals.detectors.base import BaseDetector


class OversoldDetector(BaseDetector):
    name = "oversold"

    def detect(self, df: pd.DataFrame, i: int) -> Dict[str, Any]:
        """
        触发条件：
          1. 近 lookback 天内自最高点跌幅 ≥ min_drop_from_high%
          2. 当日收阳线（涨幅 ≥ min_day_gain%）
          3. 当日最低不创新低（最低 > 前 lookback 日最低）
        """
        lookback = self._get("lookback", 10)
        if i < lookback + 1:
            return {"detected": False, "score": 0.0, "details": {"reason": "数据不足"}}

        close = df["close"].iloc[i]
        high_recent = df["high"].iloc[i - lookback:i + 1].max()
        drop = (close / high_recent - 1.0) * 100.0

        cond_drop = drop <= -self._get("min_drop_from_high", 15.0)
        if not cond_drop:
            return {"detected": False, "score": 0.0,
                    "details": {"reason": f"跌幅不足 ({drop:.1f}%)"}}

        day_ret = df["pct_chg"].iloc[i]
        cond_bull = day_ret >= self._get("min_day_gain", 1.0)
        if not cond_bull:
            return {"detected": False, "score": 0.0,
                    "details": {"reason": f"非阳线 ({day_ret:.2f}%)"}}

        if self._get("no_new_low", True):
            prev_low = df["low"].iloc[i - lookback:i].min()
            if df["low"].iloc[i] <= prev_low:
                return {"detected": False, "score": 0.0,
                        "details": {"reason": "创新低"}}

        shape_score = min(100.0, (abs(drop) - 15.0) * 3.0 + day_ret * 20.0 + 50.0)

        return {
            "detected": True,
            "score": round(shape_score, 2),
            "details": {
                "drop_pct": round(drop, 2),
                "day_ret": round(day_ret, 2),
                "high_recent": round(high_recent, 2),
            },
        }
