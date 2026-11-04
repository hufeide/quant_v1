"""强者恒强检测器：动量 / 买龙头"""
from typing import Dict, Any
import pandas as pd
from src.detectors.base import BaseDetector


class MomentumDetector(BaseDetector):
    name = "momentum"

    def detect(self, df: pd.DataFrame, i: int) -> Dict[str, Any]:
        """
        触发条件：
          1. 近 lookback 日涨幅 ≥ min_gain%
          2. 距区间高点回撤 < max_drawdown_from_high%
          3. 收盘价守住 MA_hold（close ≥ MA）
        """
        lookback = self._get("lookback", 20)
        ma_n = self._get("ma_hold", 10)
        if i < lookback or pd.isna(df[f"ma{ma_n}"].iloc[i]):
            return {"detected": False, "score": 0.0, "details": {"reason": "数据不足"}}

        close = df["close"].iloc[i]
        ret = df[f"ret_{lookback}"].iloc[i]
        if pd.isna(ret):
            return {"detected": False, "score": 0.0, "details": {"reason": "涨幅数据缺失"}}
        ret_pct = ret * 100.0

        cond_gain = ret_pct >= self._get("min_gain", 25.0)
        if not cond_gain:
            return {"detected": False, "score": 0.0,
                    "details": {"reason": f"涨幅不足 ({ret_pct:.1f}%)"}}

        high_n = df["high"].iloc[i - lookback + 1:i + 1].max()
        dd = (close / high_n - 1.0) * 100.0
        cond_dd = dd > -self._get("max_drawdown_from_high", 8.0)
        if not cond_dd:
            return {"detected": False, "score": 0.0,
                    "details": {"reason": f"回撤过大 ({dd:.1f}%)"}}

        ma_val = df[f"ma{ma_n}"].iloc[i]
        cond_ma = close >= ma_val
        if not cond_ma:
            return {"detected": False, "score": 0.0,
                    "details": {"reason": f"跌破MA{ma_n}"}}

        shape_score = min(100.0, ret_pct * 2.5 + (1.0 + dd) * 20.0 + 30.0)

        return {
            "detected": True,
            "score": round(shape_score, 2),
            "details": {
                "ret_pct": round(ret_pct, 2),
                "drawdown": round(dd, 2),
                "ma": round(ma_val, 2),
            },
        }
