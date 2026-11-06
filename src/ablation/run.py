#!/usr/bin/env python3
"""Leave-One-Out 消融实验运行器（数据驱动）

自动枚举 config.yaml 开关表中的叶子开关（modules.* 和 universe.filters.*），
对每个开关生成"关闭该项、其余全部开启"的变体配置，逐个回测对比贡献度。

不再手写实验函数——消融清单完全由 config.yaml 的开关表决定。
新增开关项只要在 yml 里声明，上层脚本自动捕获它。

用法：
    python3 src/ablation/run.py baseline      # 只跑基线（全开）
    python3 src/ablation/run.py               # 全部叶子开关
    python3 src/ablation/run.py detectors scoring risk   # 只跑指定模块的叶子
    python3 src/ablation/run.py all --out DIR # 指定输出目录

结果：
    backtest/results/ablation/ablation_comparison.csv  # 对比总表
    backtest/results/ablation/ablation_<实验名>/        # 每次实验的净值/成交
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, set_override, clear_override
from src.ablation.engine import AblationEngine, AblationResult

RESULTS_DIR = "backtest/results/ablation"


# ---------------------------------------------------------------- 开关发现

# 信号相关前缀：这些开关的改变会影响信号生成，必须重新生成信号
SIGNAL_AFFECTING_PREFIXES = ("modules.detectors", "modules.scoring",
                             "universe.filters")


def _is_signal_affecting(path: str) -> bool:
    """该开关是否影响信号生成（影响则必须重新生成信号）"""
    return path.startswith(SIGNAL_AFFECTING_PREFIXES)


def discover_switches(cfg: dict) -> list:
    """枚举开关表中的叶子开关

    返回 [{'path': 'modules.risk.rules.R1_stop_loss', 'module': 'risk', 'desc': '...'}]

    modules 段的叶子由递归展开得到（modules.<mod>.<key>），模块名取二级段；
    universe.filters 段直接列叶子。
    """
    switches = []

    def walk(node: dict, prefix: str, module: str):
        for key, val in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(val, bool):
                desc = _DEFAULT_DESC.get(
                    (module, key),
                    f"关闭 {module}.{key}")
                switches.append({"path": path, "module": module, "desc": desc})
            elif isinstance(val, dict):
                walk(val, path, module)

    # modules 顶层每个段即一个模块（detectors/scoring/risk/sentiment/portfolio）
    for mod_key, mod_node in (cfg.get("modules", {}) or {}).items():
        walk(mod_node, f"modules.{mod_key}", mod_key)

    # universe.filters 段：直接列叶子 bool
    for key, val in cfg.get("universe", {}).get("filters", {}).items():
        if isinstance(val, bool):
            switches.append({
                "path": f"universe.filters.{key}",
                "module": "universe",
                "desc": _FILTER_DESC.get(key, f"关闭过滤 {key}"),
            })

    return switches


# detectors/scoring/risk/sentiment/portfolio 各叶子开关的中文名
_DEFAULT_DESC = {
    ("detectors", "enabled"): "关闭形态检测器（无信号产出）",
    ("detectors", "pullback"): "关闭缩量回踩检测器",
    ("detectors", "breakout"): "关闭平台突破检测器",
    ("detectors", "oversold"): "关闭超跌反弹检测器",
    ("detectors", "momentum"): "关闭强者恒强检测器",
    ("scoring", "enabled"): "关闭综合评分器（信号直接入选）",
    ("risk", "enabled"): "关闭卖出风控（纯持有到期）",
    ("risk", "R1_stop_loss"): "关闭 R1 止损",
    ("risk", "R2_breakeven"): "关闭 R2 保盈",
    ("risk", "R3_trailing"): "关闭 R3 移动止盈",
    ("risk", "R4_hard_profit"): "关闭 R4 硬止盈",
    ("risk", "B1_ma_exit"): "关闭 B1 跌破均线",
    ("risk", "B2_time_stop"): "关闭 B2 时间止损",
    ("risk", "B3_limit_down"): "关闭 B3 跌停顺延",
    ("sentiment", "enabled"): "关闭情绪闸门",
    ("portfolio", "enabled"): "关闭组合管理",
    ("portfolio", "holdings_limit"): "关闭持仓只数上限",
    ("portfolio", "cooldown"): "关闭卖出冷却期",
    ("portfolio", "nav_tracking"): "关闭净值追踪",
}

# universe.filters 的中文名
_FILTER_DESC = {
    "exclude_st": "不剔 ST 股",
    "price_range": "不限制股价区间",
    "market_cap_range": "不限制流通市值区间",
    "pct_chg_limit": "不限制当日涨跌幅",
    "amount_top_n": "不按成交额排序（去除流动性筛选）",
    "overheat_exclude": "不剔除换手率过热股",
}


# ---------------------------------------------------------------- 变体生成

def _nest(parts: list, value) -> dict:
    """把 [a, b, c] 路径嵌套成 {a: {b: {c: value}}}" """
    d = {}
    node = d
    for p in parts[:-1]:
        node[p] = {}
        node = node[p]
    node[parts[-1]] = value
    return d


def build_off_patch(switch: dict) -> dict:
    """生成"关闭这个开关"的 override patch

    universe.filters.* 开关读 universe.filters.<key>，直接覆盖即可；
    modules.<mod>.<key> 开关由 module_enabled 查找（ablation.<mod>.<key>
    优先于 modules.<mod>.<key>），所以向 ablation 段注入。
    """
    path = switch["path"]
    if path.startswith("modules."):
        return _nest(["ablation"] + path[len("modules."):].split("."), False)
    return _nest(path.split("."), False)


def exp_name_of(switch: dict) -> str:
    return switch["path"].replace(".", "_") + "_off"


# ---------------------------------------------------------------- 回测执行

def run_once(cfg: dict, exp_name: str, skip_signals: bool = False) -> AblationResult:
    """用给定配置跑一次完整流程：信号生成 -> 回测 -> 汇总指标

    exp_name 为 "baseline" 时写入基线信号目录（供其他实验复用），
    其他实验名走隔离目录，避免并行进程互相覆盖 all_signals.parquet。

    skip_signals=True（开关不影响候选池/检测器/评分器时）：
    直接复用基线信号，跳过信号生成，省下大量耗时。
    """
    from src.signals.signal_engine import generate_signals
    from src.backtest.engine import load_sentiment, load_signals, run_arena
    from src.data_loader import list_codes, load_daily

    result = AblationResult(name=exp_name, description="", module="", toggled={})

    # 基线信号目录（共享）
    base_sig_dir = Path(RESULTS_DIR) / "_baseline_signals"
    if exp_name == "baseline":
        base_sig_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(cfg, paths=dict(cfg["paths"], signals=str(base_sig_dir)))
    else:
        iso = Path(RESULTS_DIR) / "_sig_isolated" / exp_name
        iso.mkdir(parents=True, exist_ok=True)
        cfg = dict(cfg, paths=dict(cfg["paths"], signals=str(iso)))

    codes = list_codes(cfg["paths"]["daily"])
    if not codes:
        result.metrics["final_return"] = 0.0
        return result

    # 信号：skip_signals 时复用基线信号，否则重新生成
    if skip_signals:
        sig_dict = load_signals(str(base_sig_dir))
        result.signal_count = sum(len(g) for g in sig_dict.values()) if sig_dict else 0
    else:
        df = load_daily(codes[0])
        run_dates = sorted(df["date"].astype(str).unique().tolist())
        signals = generate_signals(run_dates, codes=codes, cfg=cfg)
        result.signal_count = len(signals)
        sig_dict = {} if signals.empty else {d: g for d, g in signals.groupby("date")}

    sentiment = load_sentiment(cfg["paths"]["sentiment"])

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


# ---------------------------------------------------------------- 基线 & 批量

def run_baseline() -> AblationResult:
    clear_override()
    r = run_once(load_config(), "baseline")
    r.description = "基线：全部开关开启"
    r.module = "baseline"
    return r


def run_experiments(switches, out_dir: str = None) -> list:
    if out_dir is None:
        out_dir = RESULTS_DIR
    engine = AblationEngine(load_config(), results_dir=out_dir)

    results = [run_baseline()]
    engine._save_result(results[0])   # 保存基线结果（保留 nav 供对比）
    base_ret = results[0].metrics["final_return"]

    print(f"\n消融实验 {len(switches)} 项（基线收益 {base_ret:+.2f}%）\n")
    for i, sw in enumerate(switches, 1):
        t0 = time.time()
        clear_override()
        cfg = set_override(build_off_patch(sw))
        skip = not _is_signal_affecting(sw["path"])
        r = run_once(cfg, exp_name_of(sw), skip_signals=skip)
        r.description = sw["desc"]
        r.module = sw["module"]
        r.toggled = {sw["path"]: False}

        engine._save_result(r)
        delta = r.metrics["final_return"] - base_ret
        print(f"[{i:2d}/{len(switches)}] {exp_name_of(sw):50s} "
              f"收益 {r.metrics['final_return']:+6.2f}% "
              f"Δ {delta:+6.2f}%  信号 {r.signal_count:3d}  成交 {r.metrics['total_trades']:3d}  "
              f"({time.time()-t0:.0f}s)")
        results.append(r)

    comp = engine.compare(results, "baseline")
    print("\n" + "=" * 70)
    print(comp.to_string(index=False))
    print(f"\n结果已写入 {out_dir}/ablation_comparison.csv")
    return results


def main():
    ap = argparse.ArgumentParser(description="Leave-One-Out 消融实验（数据驱动）")
    ap.add_argument("module", nargs="*", help="只跑指定模块的叶子开关（如 detectors risk；也支持 universe）")
    ap.add_argument("--out", type=str, help="结果输出目录")
    args = ap.parse_args()

    cfg = load_config()

    # baseline: 只跑基线
    if set(args.module) & {"baseline"}:
        r = run_baseline()
        print(f"基线最终收益 {r.metrics['final_return']:+.2f}%")
        return

    switches = discover_switches(cfg)

    # 按模块过滤（支持 key 或中文名）
    module_names = {
        "detectors": "形态检测器",
        "scoring": "综合评分器",
        "risk": "卖出风控",
        "sentiment": "情绪闸门",
        "portfolio": "组合管理",
        "universe": "股票池过滤",
    }
    if args.module:
        wanted = set()
        for m in args.module:
            for k, v in module_names.items():
                if m in (k, v):
                    wanted.add(k)
        switches = [s for s in switches if s["module"] in wanted]

    print(f"发现 {len(switches)} 个叶子开关:")
    for s in switches:
        print(f"  [{s['module']:10s}] {s['path']:42s} {s['desc']}")

    out = args.out or RESULTS_DIR
    run_experiments(switches, out)


if __name__ == "__main__":
    main()
