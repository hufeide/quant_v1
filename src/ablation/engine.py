"""消融实验引擎：单次结果容器 + 批量对比。

消融实验的变体生成已迁移到 src/ablation/run.py（数据驱动，
只读 config.yaml 的开关表）。本模块只负责：
  - 保存单次实验结果（metrics/净值/成交）
  - 汇总所有实验，与基线对比输出贡献度表
"""
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


class AblationResult:
    """单次消融实验的结果"""

    def __init__(self, name: str, description: str = "", module: str = "",
                 toggled: Dict = None):
        self.name = name
        self.description = description
        self.module = module
        self.toggled = toggled or {}
        self.metrics: Dict[str, float] = {}
        self.nav_series: List[float] = []
        self.trade_log = None      # pd.DataFrame
        self.signal_count = 0

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "module": self.module,
            "description": self.description,
            "toggled": self.toggled,
            "metrics": self.metrics,
            "signal_count": self.signal_count,
        }


class AblationEngine:
    """消融实验引擎：保存 + 对比"""

    def __init__(self, baseline_cfg: dict, results_dir: str = "backtest/results"):
        self.baseline_cfg = baseline_cfg
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _save_result(self, result) -> None:
        """保存单次实验结果"""
        safe_name = result.name.replace("/", "_").replace(" ", "_")
        out_dir = self.results_dir / f"ablation_{safe_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(result.summary(), f, indent=2, default=str)

        if result.nav_series:
            pd.DataFrame({"nav": result.nav_series}).to_parquet(
                out_dir / "nav.parquet", index=False)

        if result.trade_log is not None and not result.trade_log.empty:
            result.trade_log.to_parquet(out_dir / "trades.parquet", index=False)

    def compare(self, results: List["AblationResult"], baseline_name: str = "baseline") -> pd.DataFrame:
        """对比所有实验与基线，输出贡献度表格"""
        baseline = next((r for r in results if r.name == baseline_name), results[0])
        baseline_ret = baseline.metrics.get("final_return", 0.0)

        rows = []
        for r in results:
            has_nav = bool(r.nav_series)
            ret = r.metrics.get("final_return", 0.0)
            diff = ret - baseline_ret
            rows.append({
                "实验": r.name,
                "模块": r.module,
                "说明": r.description,
                "信号数": r.signal_count,
                "最终收益%": round(ret, 2) if has_nav else "不可计算",
                "相对基线%": round(diff, 2) if has_nav else "不可计算",
                "贡献度": "基线" if r.name == baseline_name else ("正贡献" if diff > 0 else "负贡献"),
                "备注": "" if has_nav else "净值追踪关闭，无法计算收益",
            })

        df = pd.DataFrame(rows)
        # 排序键：收益为数值按数值排，不可计算排最后
        sort_keys = df["最终收益%"].apply(lambda v: (0, float(v)) if isinstance(v, float) else (1, 0.0))
        df["_sort"] = sort_keys
        df = df.sort_values(["_sort", "相对基线%"], ascending=[True, False]).drop(columns="_sort")
        df.to_csv(self.results_dir / "ablation_comparison.csv", index=False)
        return df
