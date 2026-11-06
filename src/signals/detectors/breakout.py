"""平台突破检测器：追强 / 趋势启动"""
from typing import Dict, Any
import pandas as pd
from src.signals.detectors.base import BaseDetector


class BreakoutDetector(BaseDetector):
    name = "breakout"

    def detect(self, df: pd.DataFrame, i: int) -> Dict[str, Any]:
        """
        触发条件：
          1. 收盘价创 lookback 日新高
          2. 成交量 ≥ vol_ma5 × vol_ratio
          3. 距前高 ≤ max_dist_from_high%
        """
        if i < self._get("lookback", 60):
            return {"detected": False, "score": 0.0, "details": {"reason": "数据不足"}}

        close = df["close"].iloc[i]
        vol = df["volume"].iloc[i]
        vol_ma5 = df["vol_ma5"].iloc[i]
        if pd.isna(vol_ma5) or vol_ma5 <= 0:
            return {"detected": False, "score": 0.0, "details": {"reason": "均量为0"}}

        lookback = self._get("lookback", 60)
        prev_high = df["high"].iloc[i - lookback:i].max()  # 不含当日

        cond_new_high = close > prev_high
        cond_vol = (vol / vol_ma5) >= self._get("vol_ratio", 1.5)
        dist = (close / prev_high - 1.0) * 100.0 if prev_high > 0 else 0.0
        cond_dist = dist <= self._get("max_dist_from_high", 6.0)

        if not cond_new_high:
            return {"detected": False, "score": 0.0, "details": {"reason": "未创新高"}}
        if not cond_vol:
            return {"detected": False, "score": 0.0,
                    "details": {"reason": f"量不足 (ratio={vol/vol_ma5:.2f})"}}
        if not cond_dist:
            return {"detected": False, "score": 0.0,
                    "details": {"reason": f"距前高太远 ({dist:.1f}%)"}}

        vol_ratio_actual = vol / vol_ma5
        shape_score = min(100.0, (vol_ratio_actual - 1.0) * 60.0 + 40.0)

        return {
            "detected": True,
            "score": round(shape_score, 2),
            "details": {
                "vol_ratio": round(vol_ratio_actual, 2),
                "dist_pct": round(dist, 2),
                "prev_high": round(prev_high, 2),
            },
        }
