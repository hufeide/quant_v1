"""综合打分器：6 项加权，满分 100"""
from typing import Dict, Any
import pandas as pd
from src.signals.detectors.base import BaseDetector


class WeightedScorer:
    """六项因子加权打分

    因子：
      1. pattern_fit (30%) - 形态契合度
      2. ma_trend (20%) - 均线趋势
      3. vol_health (15%) - 缩量健康度
      4. relative_strength (15%) - 相对强度（横截面）
      5. pullback_position (10%) - 回踩位置
      6. macd (10%) - MACD 状态
    """

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        total = sum(weights.values())
        if abs(total - 100.0) > 0.1:
            raise ValueError(f"权重之和须为 100，当前为 {total}")

    def score(self, df: pd.DataFrame, i: int, detector_result: Dict,
              relative_rank: float = 0.5) -> float:
        """
        计算综合得分

        Args:
            df: 日线数据（已含指标）
            i: 行索引
            detector_result: 检测器返回结果 {"detected": bool, "score": float, ...}
            relative_rank: 相对强度百分位 [0, 1]，由横截面排序算出

        Returns:
            综合得分 [0, 100]
        """
        if not detector_result.get("detected"):
            return 0.0

        pattern_fit = detector_result.get("score", 0.0)

        # MA 趋势打分
        ma_trend = self._score_ma_trend(df, i)

        # 缩量健康度
        vol_health = self._score_vol_health(df, i)

        # 相对强度（由外部传入百分位）
        rs_score = relative_rank * 100.0

        # 回踩位置（距 MA20 贴合度）
        pos_score = self._score_pullback_position(df, i)

        # MACD
        macd_score = self._score_macd(df, i)

        total = (
            self.weights["pattern_fit"] * pattern_fit / 100.0 +
            self.weights["ma_trend"] * ma_trend / 100.0 +
            self.weights["vol_health"] * vol_health / 100.0 +
            self.weights["relative_strength"] * rs_score / 100.0 +
            self.weights["pullback_position"] * pos_score / 100.0 +
            self.weights["macd"] * macd_score / 100.0
        )

        return round(total, 2)

    def _score_ma_trend(self, df: pd.DataFrame, i: int) -> float:
        """均线多头排列打分"""
        if i < 60 or pd.isna(df["ma20"].iloc[i]):
            return 50.0

        ma5 = df["ma5"].iloc[i]
        ma10 = df["ma10"].iloc[i]
        ma20 = df["ma20"].iloc[i]

        if ma5 > ma10 > ma20:
            return 100.0
        elif ma5 < ma10 < ma20:
            return 20.0
        else:
            # 部分多头
            if ma5 > ma10:
                return 70.0
            else:
                return 40.0

    def _score_vol_health(self, df: pd.DataFrame, i: int) -> float:
        """缩量健康度：回调期缩量越多越好"""
        if i < 20:
            return 50.0
        vol_ma5 = df["vol_ma5"].iloc[i]
        if pd.isna(vol_ma5) or vol_ma5 <= 0:
            return 50.0
        vol_ma10 = df["vol_ma10"].iloc[i]
        if pd.isna(vol_ma10) or vol_ma10 <= 0:
            return 50.0

        ratio = vol_ma5 / vol_ma10
        # ratio < 0.75 得高分，> 1.25 得低分
        if ratio < 0.75:
            return 100.0
        elif ratio > 1.25:
            return 30.0
        else:
            return 60.0

    def _score_pullback_position(self, df: pd.DataFrame, i: int) -> float:
        """回踩位置：贴近 MA20 得分高"""
        if i < 20 or pd.isna(df["ma20"].iloc[i]):
            return 50.0

        close = df["close"].iloc[i]
        ma20 = df["ma20"].iloc[i]
        dist = abs(close / ma20 - 1.0) * 100.0

        if dist < 2.0:
            return 100.0
        elif dist < 5.0:
            return 70.0
        elif dist < 10.0:
            return 50.0
        else:
            return 30.0

    def _score_macd(self, df: pd.DataFrame, i: int) -> float:
        """MACD 状态：DIF > 0 且 DEA 向上得分高"""
        if i < 30 or pd.isna(df["dif"].iloc[i]) or pd.isna(df["dea"].iloc[i]):
            return 50.0

        dif = df["dif"].iloc[i]
        dea = df["dea"].iloc[i]
        hist = df["hist"].iloc[i]

        if dif > 0 and dea > 0 and hist > 0:
            return 100.0
        elif dif > 0 and dea > 0 and hist <= 0:
            return 70.0
        elif dif < 0 and dea < 0 and hist < 0:
            return 20.0
        else:
            return 50.0
