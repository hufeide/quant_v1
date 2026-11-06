"""缩量回踩检测器：低吸 / 趋势中继"""
from typing import Dict, Any
import pandas as pd
from src.signals.detectors.base import BaseDetector
from src.indicators import detect_uptrend_and_pullback


class PullbackDetector(BaseDetector):
    name = "pullback"

    def detect(self, df: pd.DataFrame, i: int) -> Dict[str, Any]:
        """
        触发条件：
          1. 此前一波上涨，区间涨幅 ≥ min_prior_gain%
          2. 随后回调 2~8 天（含当日）
          3. 回调期间日均成交量 < 上涨期日均量 × vol_shrink_ratio
          4. 回调最低点不破 MA_support
        """
        if i < 60:
            return {"detected": False, "score": 0.0, "details": {"reason": "数据不足"}}

        diag = detect_uptrend_and_pullback(df, i, self.cfg)
        if not diag.get("ok"):
            return {"detected": False, "score": diag.get("score", 0.0),
                    "details": diag.get("reason", "未命中")}

        # 缩量校验
        vol_ok = diag["vol_ratio"] < self._get("vol_shrink_ratio", 0.75)
        if not vol_ok:
            return {"detected": False, "score": diag["score"],
                    "details": f"未缩量 (ratio={diag['vol_ratio']:.2f})"}

        # 形态契合度打分（满分 100）
        seg_ret = diag["seg_ret"]
        shrink_depth = 1.0 - diag["vol_ratio"]  # 缩得越多分越高
        shape_score = min(100.0, seg_ret * 4.0 + shrink_depth * 100.0)

        return {
            "detected": True,
            "score": round(shape_score, 2),
            "details": {
                "seg_ret": round(seg_ret, 2),
                "pb_days": diag["pb_days"],
                "vol_ratio": round(diag["vol_ratio"], 3),
                "drawdown": round(diag["drawdown"], 2),
            },
        }
