#!/usr/bin/env python3
"""子模块消融回测运行器。

对总策略的每个子模块做独立开关实验：只改一个模块，其余保持基线，
对比净值/信号数/成交数，量化各模块贡献。

用法：
    python3 src/ablation/run.py baseline     # 只跑基线
    python3 src/ablation/run.py risk         # 跑 risk 模块全部实验
    python3 src/ablation/run.py universe scoring   # 跑多个模块
    python3 src/ablation/run.py all          # 跑全部 30 项
"""
import sys
import time
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config, set_override, clear_override
from src.ablation.engine import AblationEngine, AblationResult
from src.ablation.experiments import ABLATION_EXPERIMENTS
from src.ablation.config import ABLATION_DEFAULTS

RESULTS_DIR = "backtest/results/ablation"


# ---------------------------------------------------------------- 配置映射

# 消融段 -> 策略实际读取的 config 路径
# universe 过滤器读 universe.filters.*（见 src/universe/filter.py）
PATH_MAP = {
    "universe": "universe.filters",
    "scoring": "scoring.weights",
    "risk": "ablation.risk",
    "sentiment": "ablation.sentiment",
    "position": "position",
    "buy_guards": "ablation.buy_guards",
}


def build_cfg(module: str, overrides: dict = None) -> dict:
    """在完整 config 上叠加消融开关，产出策略实际读取的配置树。

    先 clear_override 保证每次实验从干净状态开始，
    避免上一组实验的开关残留影响当前组。
    """
    clear_override()
    cfg = load_config()
    patch = {"ablation": deepcopy(ABLATION_DEFAULTS)}

    if overrides:
        # overrides 是 {module: {key: value}}，按模块写到对应位置
        for mod, kv in overrides.items():
            path = PATH_MAP.get(mod, f"ablation.{mod}")
            parts = path.split(".")
            node = patch
            for p in parts:
                node = node.setdefault(p, {})
            node.update(kv)

    return set_override(patch)


# ---------------------------------------------------------------- 回测执行

def run_once(cfg: dict, exp_name: str) -> AblationResult:
    """用给定配置跑一次完整流程：信号生成 -> 回测 -> 汇总指标

    exp_name 为 "baseline" 时写入共享信号目录（数据复用），
    其他实验名走隔离目录，避免并行进程互相覆盖 all_signals.parquet。
    """
    from src.signal_engine import generate_signals
    from bt_engine.cerebro import load_sentiment, run_arena
    from src.data_loader import list_codes, load_daily

    result = AblationResult(name=exp_name, description="", module="", toggled={})

    # 隔离信号输出目录（并行安全）
    if exp_name != "baseline":
        iso = Path(RESULTS_DIR) / "_sig_isolated" / exp_name
        iso.mkdir(parents=True, exist_ok=True)
        cfg = dict(cfg, paths=dict(cfg["paths"], signals=str(iso)))

    codes = list_codes(cfg["paths"]["daily"])
    if not codes:
        result.metrics["final_return"] = 0.0
        return result

    # 信号生成窗口：取第一只股票的全部交易日（覆盖预热 + 回测区间）
    df = load_daily(codes[0])
    run_dates = sorted(df["date"].astype(str).unique().tolist())

    signals = generate_signals(run_dates, codes=codes)
    result.signal_count = len(signals)

    sentiment = load_sentiment(cfg["paths"]["sentiment"])
    sig_dict = {}
    if not signals.empty:
        sig_dict = {d: g for d, g in signals.groupby("date")}

    strats = run_arena(
        codes=codes,
        signals=sig_dict,
        sentiment=sentiment,
        strategies=["pullback", "breakout", "oversold", "momentum"],
        initial_cash=cfg["project"]["initial_cash"],
    )

    # 四账户净值取均值
    all_nav = []
    all_trades = []
    for s in strats.values():
        all_nav.extend(s.portfolio.daily_nav)
        all_trades.extend(s.portfolio.trades)

    result.trade_log = pd.DataFrame(all_trades) if all_trades else None
    result.metrics["total_trades"] = len(all_trades)

    if all_nav:
        nav_df = pd.DataFrame(all_nav)
        nav_mean = nav_df.groupby("date")["nav"].mean()
        result.nav_series = nav_mean.tolist()
        result.metrics["final_return"] = (nav_mean.iloc[-1] - 1) * 100
    else:
        result.metrics["final_return"] = 0.0

    result.metrics["max_drawdown"] = _max_drawdown(result.nav_series)
    result.metrics["buy_count"] = int(len([t for t in (all_trades or []) if t["side"] == "buy"]))
    result.metrics["sell_count"] = int(len([t for t in (all_trades or []) if t["side"] == "sell"]))

    return result


def _max_drawdown(nav) -> float:
    if not nav:
        return 0.0
    peak = nav[0]
    mdd = 0.0
    for v in nav:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return round(mdd * 100, 2)


# ---------------------------------------------------------------- 实验调度

def run_baseline() -> AblationResult:
    cfg = build_cfg("baseline")
    r = run_once(cfg, "baseline")
    r.description = "基线：全部子模块启用"
    r.module = "baseline"
    return r


def extract_overrides(toggle_fn) -> dict:
    """执行 toggle_fn，对比基线找出实际改动的键"""
    base = deepcopy(ABLATION_DEFAULTS)
    new = toggle_fn(base)
    out = {}
    for mod in ABLATION_DEFAULTS:
        for k in ABLATION_DEFAULTS[mod]:
            if k in new.get(mod, {}) and new[mod][k] != base[mod][k]:
                out.setdefault(mod, {})[k] = new[mod][k]
    return out


def run_experiments(modules=None, out_dir: str = None) -> list:

    if out_dir is None:
        out_dir = RESULTS_DIR
    engine = AblationEngine(load_config(), results_dir=out_dir)
    results = [run_baseline()]
    base_ret = results[0].metrics["final_return"]

    keys = list(ABLATION_EXPERIMENTS.keys())
    if modules:
        want = {f"{m}" if not m.startswith("ablation") else m for m in modules}
        keys = [k for k in keys if ABLATION_EXPERIMENTS[k][1] in want]

    print(f"\n消融实验 {len(keys)} 项（基线收益 {base_ret:+.2f}%）\n")
    for i, key in enumerate(keys, 1):
        desc, mod, toggle_fn = ABLATION_EXPERIMENTS[key]
        overrides = extract_overrides(toggle_fn)
        t0 = time.time()

        cfg = build_cfg(mod, overrides)
        r = run_once(cfg, key)
        r.description = desc
        r.module = mod
        r.toggled = overrides

        engine._save_result(r)
        delta = r.metrics["final_return"] - base_ret
        print(f"[{i:2d}/{len(keys)}] {key:34s} 收益 {r.metrics['final_return']:+6.2f}% "
              f"Δ {delta:+6.2f}%  信号 {r.signal_count:3d}  成交 {r.metrics['total_trades']:3d}  "
              f"({time.time()-t0:.0f}s)")
        results.append(r)

    comp = engine.compare(results, "baseline")
    print("\n" + "=" * 70)
    print(comp.to_string(index=False))
    print(f"\n结果已写入 {out_dir}/ablation_comparison.csv")
    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    arg = sys.argv[1]
    out_dir = None
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        out_dir = sys.argv[i + 1]
    if arg == "baseline":
        r = run_baseline()
        print(f"\n基线：收益 {r.metrics['final_return']:+.2f}%，"
              f"最大回撤 {r.metrics['max_drawdown']:.2f}%，"
              f"信号 {r.signal_count}，成交 {r.metrics['total_trades']}")
    elif arg == "all":
        run_experiments(out_dir=out_dir)
    else:
        run_experiments(arg.split(), out_dir=out_dir)


if __name__ == "__main__":
    main()
