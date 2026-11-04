#!/usr/bin/env python3
"""生成策略竞技场看板 HTML。

读取回测结果 + config.yaml，产出单文件 dashboard.html
（数据内嵌、图表为内联 SVG、原生 JS 提供悬停/键盘提示）。

用法：python3 web/build_dashboard.py
"""
import json
import sys
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "backtest" / "results"
SIGNALS = ROOT / "data" / "signals"
SNAPSHOT = ROOT / "data" / "snapshot"
SENTIMENT = ROOT / "data" / "sentiment"

STRATEGY_META = {
    "pullback":  {"account": "A1", "cn": "缩量回踩", "logic": "趋势中继 · 低吸"},
    "breakout":  {"account": "A2", "cn": "平台突破", "logic": "趋势启动 · 追强"},
    "oversold":  {"account": "A3", "cn": "超跌反弹", "logic": "均值回归 · 抄底"},
    "momentum":  {"account": "A4", "cn": "强者恒强", "logic": "动量 · 买龙头"},
}
SLOT = {"pullback": 0, "breakout": 1, "oversold": 2, "momentum": 3}

# 卖出规则代号，按 priority 从高到低（与 src/risk/rules.py PRIORITY 一致）
RULE_ORDER = ["R4", "R3", "R2", "R1", "B1", "B2", "B3"]

# rules.py 中各规则 reason 模板的标志性前缀（reason 只存中文描述，不带规则代号）
RULE_PREFIXES = [
    ("R4", "硬止盈"),
    ("R3", "移动止盈"),
    ("R2", "保盈"),
    ("R2", "回落成本"),
    ("R1", "止损触发"),
    ("B1", "有效跌破MA"),
    ("B2", "时间止损"),
    ("B3", "跌停封死"),
]


def classify_reason(txt: str) -> str:
    """把 rules.py 的中文 reason 描述归类到规则代号；未命中返回 '持有中'"""
    for code, prefix in RULE_PREFIXES:
        if txt.startswith(prefix):
            return code
    return "持有中"


def load_cfg():
    """复用 src.config 的加载器，拿到数值化后的配置"""
    from src.config import load_config
    return load_config(str(ROOT / "config.yaml"))


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_all():
    cfg = load_cfg()
    summary = read_json(RESULTS / "summary.json")
    trades = pd.read_parquet(RESULTS / "trades.parquet")
    nav = pd.read_parquet(RESULTS / "daily_nav.parquet")
    signals = pd.read_parquet(SIGNALS / "all_signals.parquet")
    sentiment = pd.read_parquet(SENTIMENT / "sentiment.parquet")
    return cfg, summary, trades, nav, signals, sentiment


def build_dataset(cfg, summary, trades, nav, signals, sentiment):
    """汇总看板需要的全部数值"""
    ds = {}

    ds["cfg"] = cfg
    ds["strategies"] = []
    for name, meta in STRATEGY_META.items():
        s = summary.get(name, {})
        sub = trades[trades["strategy"] == name] if not trades.empty else trades
        sells = sub[sub["side"] == "sell"]
        ds["strategies"].append({
            "name": name, "account": meta["account"],
            "cn": meta["cn"], "logic": meta["logic"], "slot": SLOT[name],
            "final_nav": s.get("final_nav"),
            "return_pct": s.get("total_return_pct"),
            "trades": s.get("total_trades", 0),
            "buys": s.get("buys", 0), "sells": s.get("sells", 0),
            "win_rate": s.get("win_rate_pct"),
            "sell_rows": sells.to_dict("records"),
        })

    # 净值序列
    nav["date"] = nav["date"].astype(str)
    ds["nav"] = nav.to_dict("records")
    ds["nav_dates"] = sorted(nav["date"].unique().tolist())

    # 卖出原因分布（全竞技场）
    # rules.py 的 reason 字段是中文描述（如 "有效跌破MA20"），不带规则代号前缀，
    # 这里按各规则 reason 模板的标志性前缀归类（见 RULE_PREFIXES）。
    sell_all = trades[trades["side"] == "sell"] if not trades.empty else trades
    reason_count = {}
    for raw in sell_all["reason"]:
        txt = str(raw).strip() if pd.notna(raw) else "持有中"
        rule = classify_reason(txt)
        reason_count[rule] = reason_count.get(rule, 0) + 1
    ds["reasons"] = [{"rule": o, "count": reason_count.get(o, 0)} for o in RULE_ORDER]
    ds["reason_detail"] = {k: v for k, v in sorted(reason_count.items())}
    ds["reason_total"] = len(sell_all)

    # 信号分布
    sig_by_strat = signals.groupby("detector").size().to_dict() if not signals.empty else {}
    ds["signals"] = [
        {"name": n, "cn": STRATEGY_META[n]["cn"], "slot": SLOT[n],
         "count": int(sig_by_strat.get(n, 0))}
        for n in STRATEGY_META
    ]
    ds["signal_total"] = int(sum(sig_by_strat.values()))

    # 候选池漏斗
    snap_files = sorted(SNAPSHOT.glob("*.parquet"))
    funnel = {"total_snapshots": len(snap_files)}
    if snap_files:
        snap = pd.concat([pd.read_parquet(p) for p in snap_files])
        snap["market_cap_float"] = snap["market_cap_float"].astype(float)
        snap["close"] = snap["close"].astype(float)
        snap["pct_chg"] = snap["pct_chg"].astype(float)
        u = cfg["universe"]
        funnel["raw_rows"] = int(len(snap))
        funnel["distinct_codes"] = int(snap["code"].nunique())
        funnel["pass_price"] = int((snap["close"].between(u["min_price"], u["max_price"])).sum())
        funnel["pass_cap"] = int((snap["market_cap_float"].between(
            u["min_float_mv"], u["max_float_mv"])).sum())
        funnel["pass_pct"] = int((snap["pct_chg"].between(
            u["max_daily_pct_down"], u["max_daily_pct_up"])).sum())
        funnel["candidate_pool"] = int(u["candidate_pool_size"])
        ds["universe"] = funnel
        ds["universe_days"] = len(snap_files)
    else:
        ds["universe"] = funnel
        ds["universe_days"] = 0

    # 情绪指数
    sentiment = sentiment.copy()
    sentiment["date"] = sentiment["date"].astype(str)
    ds["sentiment"] = sentiment.to_dict("records")
    gate = cfg["sentiment"]
    ds["gate"] = gate
    ds["gate_bands"] = [
        {"band": "开放开仓", "low": gate["gate_limit"], "high": 100.0,
         "mode": "open", "max_buy": int(cfg["position"]["max_buy_per_round"])},
        {"band": "限开 1 只", "low": gate["gate_limit"], "high": gate["gate_open"],
         "mode": "limited", "max_buy": 1},
        {"band": "暂停开仓", "low": 0.0, "high": gate["gate_limit"],
         "mode": "halted", "max_buy": 0},
    ]

    # 形态阈值（子模块数值）
    det = cfg["detectors"]
    ds["detectors"] = {
        "pullback": {"cn": "缩量回踩", "thresholds": [
            ["上涨波段涨幅", f">= {det['pullback']['min_prior_gain']}%"],
            ["回调天数", f"{det['pullback']['min_pullback_days']}~{det['pullback']['max_pullback_days']} 天"],
            ["缩量比例", f"< 上涨期均量 × {det['pullback']['vol_shrink_ratio']}"],
            ["均线支撑", f"不破 MA{det['pullback']['ma_support']}"],
        ]},
        "breakout": {"cn": "平台突破", "thresholds": [
            ["创新高", f"创 {det['breakout']['lookback']} 日新高"],
            ["量比", f">= 5日均量 × {det['breakout']['vol_ratio']}"],
            ["距前高", f"<= {det['breakout']['max_dist_from_high']}%"],
            ["停用日期", str(det['breakout'].get('run_until', '—'))],
        ]},
        "oversold": {"cn": "超跌反弹", "thresholds": [
            ["回看窗口", f"{det['oversold']['lookback']} 日"],
            ["区间跌幅", f">= {det['oversold']['min_drop_from_high']}%"],
            ["当日阳线", f"涨幅 >= {det['oversold']['min_day_gain']}%"],
            ["不创新低", "当日最低 > 前窗口最低"],
        ]},
        "momentum": {"cn": "强者恒强", "thresholds": [
            ["回看窗口", f"{det['momentum']['lookback']} 日"],
            ["区间涨幅", f">= {det['momentum']['min_gain']}%"],
            ["距高点回撤", f"< {det['momentum']['max_drawdown_from_high']}%"],
            ["均线守住", f"收盘 >= MA{det['momentum']['ma_hold']}"],
        ]},
    }
    return ds


HTML_TEMPLATE = ROOT / "web" / "dashboard_template.html"


def sanitize(obj):
    """把 NaN/Inf 转成 None。

    pandas 的 to_dict("records") 保留 float('nan')，json.dumps 会输出非法的
    `NaN` 字面量（非 JSON 标准），浏览器 JSON.parse 直接报错。
    """
    import math
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def main():
    cfg, summary, trades, nav, signals, sentiment = load_all()
    ds = build_dataset(cfg, summary, trades, nav, signals, sentiment)
    ds["backtest_window"] = {
        "start": cfg["project"]["backtest_start"],
        "end": cfg["project"]["backtest_end"],
        "initial_cash": cfg["project"]["initial_cash"],
    }
    ds["cfg"] = {k: v for k, v in cfg.items() if k != "paths"}

    payload = json.dumps(sanitize(ds), ensure_ascii=False, default=str)
    html = HTML_TEMPLATE.read_text(encoding="utf-8")
    out = ROOT / "dashboard.html"
    out.write_text(html.replace("__DATA_JSON__", payload), encoding="utf-8")
    print(f"✓ dashboard.html 已生成  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
