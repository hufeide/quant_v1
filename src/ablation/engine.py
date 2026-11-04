"""消融回测引擎：子模块独立开关 + 贡献分析。

设计理念：
  - 不改策略代码，通过 config override 注入开关
  - 每条消融实验独立跑回测，记录净值、成交、信号质量
  - 与基线对比，量化每个子模块的贡献
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np


@dataclass
class AblationResult:
    """单次消融实验的结果"""
    name: str
    description: str
    module: str                    # universe/scoring/risk/sentiment/position/buy_guards
    toggled: Dict                    # 实际改动的配置项
    metrics: Dict[str, float] = field(default_factory=dict)
    nav_series: List[float] = field(default_factory=list)
    trade_log: pd.DataFrame = None
    signal_count: int = 0

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
    """消融回测引擎"""

    def __init__(self, baseline_cfg: dict, results_dir: str = "backtest/results"):
        self.baseline_cfg = baseline_cfg
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(
        self,
        name: str,
        description: str,
        module: str,
        toggle_fn: Callable[[dict], dict],
        run_backtest_fn: Callable[[dict], AblationResult],
    ) -> AblationResult:
        """
        跑一次消融实验

        Args:
            name: 实验名（用于文件名）
            description: 人类可读的描述
            module: 所属模块
            toggle_fn: 接收 baseline_cfg，返回 override 后的配置
            run_backtest_fn: 接收 override 后的配置，跑回测并返回 AblationResult
        """
        overridden_cfg = toggle_fn(self.baseline_cfg)
        result = run_backtest_fn(overridden_cfg)
        result.name = name
        result.description = description
        result.module = module
        result.toggled = {
            k: v for k, v in overridden_cfg.get("ablation", {}).items()
            if k in self.baseline_cfg.get("ablation", {}) and v != self.baseline_cfg["ablation"][k]
        }

        # 保存结果
        self._save_result(result)
        print(f"[消融] {name}: 信号 {result.signal_count}, 净值 {result.metrics.get('final_return', 0):+.2f}%")
        return result

    def _save_result(self, result: AblationResult) -> None:
        """保存单次实验结果"""
        safe_name = result.name.replace("/", "_").replace(" ", "_")
        out_dir = self.results_dir / f"ablation_{safe_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 指标
        import json
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(result.summary(), f, indent=2, default=str)

        # 净值
        if result.nav_series:
            pd.DataFrame({"nav": result.nav_series}).to_parquet(
                out_dir / "nav.parquet", index=False
            )

        # 成交
        if result.trade_log is not None and not result.trade_log.empty:
            result.trade_log.to_parquet(out_dir / "trades.parquet", index=False)

    def run_all(self, experiments: List[Dict]) -> List[AblationResult]:
        """批量跑实验"""
        results = []
        for exp in experiments:
            res = self.run_experiment(
                name=exp["name"],
                description=exp["description"],
                module=exp["module"],
                toggle_fn=exp["toggle_fn"],
                run_backtest_fn=exp["run_fn"],
            )
            results.append(res)
        return results

    def compare(self, results: List[AblationResult], baseline_name: str = "baseline") -> pd.DataFrame:
        """对比所有实验与基线，输出贡献度表格"""
        baseline = next((r for r in results if r.name == baseline_name), results[0])
        baseline_ret = baseline.metrics.get("final_return", 0.0)

        rows = []
        for r in results:
            ret = r.metrics.get("final_return", 0.0)
            diff = ret - baseline_ret
            rows.append({
                "实验": r.name,
                "模块": r.module,
                "信号数": r.signal_count,
                "最终收益%": round(ret, 2),
                "相对基线%": round(diff, 2),
                "贡献度": "基线" if r.name == baseline_name else ("正贡献" if diff > 0 else "负贡献"),
            })

        df = pd.DataFrame(rows).sort_values("最终收益%", ascending=False)
        df.to_csv(self.results_dir / "ablation_comparison.csv", index=False)
        return df
