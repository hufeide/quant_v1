#!/usr/bin/env python3
"""策略回测控制台后端（零依赖，纯标准库 http.server）。

前端 index.html 与后端同目录。提供 API：

  GET  /                     -> index.html
  GET  /api/config           -> 当前 config.yaml 全量（前端渲染初始表单）
  GET  /api/system           -> 数据规模等系统信息
  POST /api/run             body {"patch": {…}} -> 异步启动回测，返回 run_id
  GET  /api/run/<id>/status  -> 回测进度 / 结果 / 净值数组
  GET  /api/runlist          -> 全部已完成回测的摘要列表
  GET  /api/compare?ids=a,b,c -> 多回测净值叠加（按统一日期轴对齐）
  POST /api/save             body {"patch": {…}} -> 合并写回 config.yaml

用法：
  python3 web/server.py [--port 8520]
访问 http://localhost:8520
"""
import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import yaml  # noqa: E402

import src.config as _sc  # noqa: E402
from src.config import load_config, clear_override, set_override  # noqa: E402

CONFIG_PATH = ROOT / "config.yaml"
WEB_DIR = Path(__file__).resolve().parent
SIGNAL_ROOT = ROOT / "data" / "_web_run_signals"

_LOCK = threading.Lock()
_QUEUE: "queue.Queue" = queue.Queue()
JOBS: dict = {}
START_TS = time.time()

STRATEGIES = ["pullback", "breakout", "oversold", "momentum"]
ACCOUNT_LABELS = {
    "pullback": "A1 缩量回踩",
    "breakout": "A2 平台突破",
    "oversold": "A3 超跌反弹",
    "momentum": "A4 强者恒强",
}
PALETTE = ["#38bdf8", "#f472b6", "#a3e635", "#fbbf24",
           "#fb7185", "#2dd4bf", "#c084fc", "#f97316"]


def new_run_id() -> str:
    return datetime.now().strftime("%m%d%H%M%S") + "-" + uuid.uuid4().hex[:4]


def _deep_merge(base: dict, patch: dict) -> dict:
    """深合并：对同 key 的 dict 递归合并，其余追加/覆盖"""
    out = {k: v for k, v in base.items()}
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_benchmark():
    """加载沪深300指数日线 [(date, close), ...]（结果缓存）。

    数据源：data/index/hs300.csv（东方财富行情提取）。
    文件缺失时返回 []，回测结果不带基准曲线。
    """
    global _BENCH_CACHE, _BENCH_CACHE_LOADED
    if not _BENCH_CACHE_LOADED:
        csv_path = ROOT / "data" / "index" / "hs300.csv"
        pts = []
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    pts.append((row["date"], float(row["close"])))
        _BENCH_CACHE = pts
        _BENCH_CACHE_LOADED = True
    return _BENCH_CACHE

_BENCH_CACHE = []
_BENCH_CACHE_LOADED = False


def run_job(run_id: str, patch: dict) -> None:
    """执行一次回测。串行跑队列任务，每次 clear_override 前后清干净"""
    job = JOBS[run_id]
    job["status"] = "running"
    job["message"] = "启动中…"
    try:
        patch = json.loads(json.dumps(patch or {}))
        # 信号隔离到本 run 目录，不动 data/signals
        sig_dir = SIGNAL_ROOT / run_id
        sig_dir.mkdir(parents=True, exist_ok=True)
        patch.setdefault("paths", {})["signals"] = str(sig_dir)

        from src.data_loader import list_codes, load_daily
        from src.signal_engine import generate_signals
        from bt_engine.cerebro import load_sentiment, run_arena

        with _LOCK:
            clear_override()
            merged = set_override(patch)
            try:
                codes = list_codes(merged["paths"]["daily"])
                if not codes:
                    raise RuntimeError("无可用日线数据，请先运行数据生成")
                df = load_daily(codes[0])
                run_dates = [str(d)[:10] for d in df["date"].unique()]
                job["message"] = f"信号生成：{len(codes)} 只 × {len(run_dates)} 日"

                signals_df = generate_signals(run_dates, codes=codes, cfg=merged)
                sig_dict = {} if signals_df.empty else {
                    str(d): g for d, g in signals_df.groupby("date")}
                job["message"] = f"信号 {len(signals_df)} 条，开始回测"

                sentiment = load_sentiment(merged["paths"]["sentiment"])
                # 按检测器开关过滤策略：只用在 build_detectors 里真正开启的检测器建账户，
                # 避免"只选一个却回测4次"。逻辑与 src/signal_engine.build_detectors 保持一致。
                modules_cfg = merged.get("modules", {}).get("detectors", {})
                leg_cfg = merged.get("detectors", {})
                enabled_strats = []
                for name in STRATEGIES:
                    on_modules = modules_cfg.get(name, True) if isinstance(modules_cfg, dict) else True
                    on_legacy = bool((leg_cfg.get(name) or {}).get("enabled", True))
                    if bool(on_modules) and on_legacy:
                        enabled_strats.append(name)
                if not enabled_strats:
                    enabled_strats = list(STRATEGIES)  # 全禁时兜底,保证有账户可展示
                strats = run_arena(
                    codes=codes, signals=sig_dict, sentiment=sentiment,
                    strategies=enabled_strats,
                    initial_cash=float(merged["project"]["initial_cash"]),
                )

                accounts, trades_all = {}, []
                for sname, strat in strats.items():
                    accounts[sname] = [
                        {"date": r["date"], "nav": r["nav"]}
                        for r in strat.portfolio.daily_nav]
                    for t in strat.portfolio.trades:
                        tt = dict(t)
                        tt["strategy"] = sname
                        trades_all.append(tt)

                sells = [t for t in trades_all if t["side"] == "sell"]
                wins = [t for t in sells if t.get("gain_pct", 0) > 0]

                # 跨账户逐日平均净值
                by_date: dict = {}
                for a in accounts.values():
                    for r in a:
                        by_date.setdefault(r["date"], []).append(r["nav"])
                dates = sorted(by_date)
                avg = [{"date": d, "nav": sum(by_date[d]) / len(by_date[d])}
                       for d in dates]

                peak, mdd = 1.0, 0.0
                for r in avg:
                    no = r["nav"]
                    if no > peak:
                        peak = no
                    if peak > 0:
                        dd = no / peak - 1
                        if dd < mdd:
                            mdd = dd

                final_nav = avg[-1]["nav"] if avg else 1.0
                summary = {
                    "final_nav": round(final_nav, 4),
                    "total_return_pct": round((final_nav - 1) * 100, 2),
                    "max_drawdown_pct": round(mdd * 100, 2),
                    "signal_count": len(signals_df),
                    "backtest_dates": len(dates),
                    "trades": len(trades_all),
                    "buys": len([t for t in trades_all if t["side"] == "buy"]),
                    "sells": len(sells),
                    "win_rate_pct": round(len(wins) / len(sells) * 100, 1) if sells else 0.0,
                    "accounts": {},
                }

                # 基准：沪深300 对齐到回测交易日，起始归一为 1.0
                bmap = {d: c for d, c in load_benchmark()}
                bench = []
                if dates and dates[0] in bmap:
                    b_first = bmap[dates[0]]
                    for d in dates:
                        c = bmap.get(d)
                        if c:
                            bench.append({"date": d, "nav": round(c / b_first, 6)})
                if bench:
                    summary["benchmark"] = {
                        "final_nav": round(bench[-1]["nav"], 4),
                        "return_pct": round((bench[-1]["nav"] - 1) * 100, 2),
                    }

                for sname, a_seq in accounts.items():
                    nav = a_seq[-1]["nav"] if a_seq else 1.0
                    summary["accounts"][sname] = {
                        "label": ACCOUNT_LABELS.get(sname, sname),
                        "nav": round(nav, 4),
                        "return_pct": round((nav - 1) * 100, 2),
                        "trades": len([t for t in trades_all if t.get("strategy") == sname]),
                    }

                job.update(
                    status="done",
                    message="完成",
                    finished=datetime.now().isoformat(timespec="seconds"),
                    summary=summary,
                    navs={"dates": dates, "accounts": accounts, "benchmark": bench},
                    trades=trades_all,
                )
            finally:
                clear_override()
    except Exception as e:
        job["status"] = "error"
        job["message"] = str(e)[:400]
        job["finished"] = datetime.now().isoformat(timespec="seconds")


def worker_loop():
    while True:
        spec = _QUEUE.get()
        run_job(spec["id"], spec["patch"])


def start_worker():
    threading.Thread(target=worker_loop, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "QuantConsole/1.0"

    def log_message(self, fmt, *args):
        pass

    # ---- 响应工具
    def _send(self, code: int, body: bytes,
              ctype: str = "application/json; charset=utf-8"):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False,
                                    default=str).encode("utf-8"))

    def _read_payload(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    # ---- GET
    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        try:
            if path in ("/", "/index.html"):
                return self._serve_static(WEB_DIR / "index.html")
            if path == "/api/config":
                return self._json(200, {"config": load_config()})
            if path == "/api/system":
                files = list((ROOT / "data" / "daily").glob("*.parquet"))
                return self._json(200, {
                    "daily_files": len(files),
                    "signal_cache": (ROOT / "data" / "signals" / "all_signals.parquet").exists(),
                    "uptime_min": round((time.time() - START_TS) / 60, 1),
                })
            if path.startswith("/api/run/"):
                parts = path.split("/")
                # ["", "api", "run", "<id>", "status"]
                if len(parts) >= 5 and parts[4] == "status":
                    run_id = parts[3]
                    if run_id in JOBS:
                        return self._json(200, JOBS[run_id])
                    return self._json(404, {"error": "未知回测id"})
            if path == "/api/runlist":
                return self._json(200, {"runs": [
                    dict(v) for v in sorted(JOBS.values(),
                                             key=lambda r: r["created"], reverse=True)]})
            if path == "/api/compare":
                ids = parse_qs(url.query).get("ids", [""])[0].split(",")
                picked = [JOBS[i] for i in ids if i in JOBS]
                picked.sort(key=lambda r: r["created"])
                all_dates = set()
                for r in picked:
                    for a in (r.get("navs") or {}).get("accounts", {}).values():
                        all_dates.update(x["date"] for x in a)
                dates = sorted(all_dates)
                index = {d: i for i, d in enumerate(dates)}
                series = []
                for r in picked:
                    acc=""
                    for i, (sname, a_seq) in enumerate((r.get("navs") or {}).get("accounts", {}).items()):
                        nav_map = {x["date"]: x["nav"] for x in a_seq}
                        series.append({
                            "run_id": r["id"],
                            "run_name": r["name"],
                            "account": sname,
                            "account_label": ACCOUNT_LABELS.get(sname, sname),
                            "name": f"{r['name']} · {sname}",
                            "color": PALETTE[i % len(PALETTE)],
                            "values": [nav_map.get(d) for d in dates],
                        })
                return self._json(200, {"dates": dates, "series": series})
            return self._json(404, {"error": "not found"})
        except Exception as e:
            return self._json(500, {"error": str(e)[:300]})

    def _serve_static(self, path: Path):
        if not path.exists():
            return self._json(404, {"error": str(path)})
        ctype = {"html": "text/html; charset=utf-8",
                 "js": "application/javascript; charset=utf-8",
                 "css": "text/css; charset=utf-8"}.get(
            path.suffix.lstrip("."), "application/octet-stream")
        self._send(200, path.read_bytes(), ctype)

    # ---- POST
    def do_POST(self):
        path = urlparse(self.path).path
        payload = self._read_payload()
        try:
            if path == "/api/run":
                run_id = new_run_id()
                JOBS[run_id] = {
                    "id": run_id,
                    "name": (payload.get("name") or "run " + run_id).strip(),
                    "status": "queued",
                    "message": "排队中…",
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "patch": payload.get("patch") or {},
                }
                _QUEUE.put({"id": run_id, "patch": payload.get("patch") or {}})
                return self._json(200, {"run_id": run_id})
            if path == "/api/save":
                merged = _deep_merge(load_config(), payload.get("patch") or {})
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    yaml.safe_dump(merged, f, allow_unicode=True,
                                        sort_keys=False, width=2000,
                                        indent=2)
                _sc._config_cache = None
                with open(CONFIG_PATH, encoding="utf-8") as f:
                    reloaded = yaml.safe_load(f.read())
                return self._json(200, {"saved": True, "config": reloaded})
            return self._json(404, {"error": "not found"})
        except Exception as e:
            return self._json(500, {"error": str(e)[:300]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8520)
    args = ap.parse_args()
    start_worker()
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"策略竞技场回测控制台 → http://localhost:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
