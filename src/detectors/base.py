"""形态检测器基类"""
from typing import Dict, Any
import pandas as pd


class BaseDetector:
    """所有形态检测器的基类"""

    name: str = "base"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def detect(self, df: pd.DataFrame, i: int) -> Dict[str, Any]:
        """
        在第 i 行检测形态

        Args:
            df: 已含指标的日线 DataFrame
            i: 行索引

        Returns:
            {"detected": bool, "score": float, "details": dict}
        """
        raise NotImplementedError

    def _get(self, key: str, default):
        return self.cfg.get(key, default)
