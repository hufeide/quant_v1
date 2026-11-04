"""信号引擎：日线扫描 → 形态检测 → 综合打分 → 候选信号输出

流程：
  1. 加载全市场日线数据
  2. 用快照做粗筛（universe filter）
  3. 对每只候选股扫描全部交易日，运行四个检测器
  4. 计算六项综合打分
  5. 按交易日汇总输出信号 DataFrame
"""
from pathlib import Path
from typing import Dict, List
import pandas as pd
from src.config import load_config, get_config
from src.data_loader import load_daily_with_indicators, list_codes
from src.universe.filter import filter_universe, load_snapshot
from src.indicators import add_indicators
from src.detectors.pullback import PullbackDetector
from src.detectors.breakout import BreakoutDetector
from src.detectors.oversold import OversoldDetector
from src.detectors.momentum import MomentumDetector
from src.scoring.weighted_scorer import WeightedScorer


def build_detectors(cfg: Dict) -> Dict:
    """构建四个检测器实例"""
    dc = cfg.get("detectors", {})
    return {
        "pullback":  PullbackDetector(dc.get("pullback", {})),
        "breakout":  BreakoutDetector(dc.get("breakout", {})),
        "oversold":  OversoldDetector(dc.get("oversold", {})),
        "momentum":  MomentumDetector(dc.get("momentum", {})),
    }


def scan_stock(code: str, detectors: Dict, scorer: WeightedScorer,
               run_dates: List[str], daily_dir: str) -> pd.DataFrame:
    """对单只股票扫描指定日期范围，输出信号行"""
    try:
        df = load_daily_with_indicators(code, daily_dir)
    except FileNotFoundError:
        return pd.DataFrame()

    if "date" not in df.columns or df.empty:
        return pd.DataFrame()

    rows = []
    date_set = set(run_dates)
    n = len(df)

    # 预算各检测器
    det_results = {}
    for dname, det in detectors.items():
        # 检查该策略是否在该日期范围运行
        dcfg = get_config(f"detectors.{dname}", {}) or {}
        run_from = dcfg.get("run_from")
        run_until = dcfg.get("run_until")
        if run_from and run_until:
            enabled_dates = [d for d in run_dates if run_from <= d <= run_until]
        elif run_from:
            enabled_dates = [d for d in run_dates if d >= run_from]
        elif run_until:
            enabled_dates = [d for d in run_dates if d <= run_until]
        else:
            enabled_dates = list(run_dates)

        if not enabled_dates:
            continue

        det_results[dname] = (det, set(enabled_dates))

    # 逐日扫描
    for i in range(n):
        date_str = str(df["date"].iloc[i])[:10]
        if date_str not in date_set:
            continue

        for dname, (det, enabled) in det_results.items():
            if date_str not in enabled:
                continue
            try:
                r = det.detect(df, i)
            except Exception:
                continue

            if not r.get("detected"):
                continue

            # 相对强度：用 ret_20 在当日全市场的百分位（简化：留 0.5）
            relative_rank = 0.5
            score = scorer.score(df, i, r, relative_rank)

            if score >= get_config("scoring.buy_max_score", 70.0):
                continue  # 超过阈值不买

            rows.append({
                "date": date_str,
                "code": code,
                "detector": dname,
                "score": score,
                "close": float(df["close"].iloc[i]),
                "pct_chg": float(df["pct_chg"].iloc[i]) if "pct_chg" in df.columns else 0.0,
                "amount": float(df["amount"].iloc[i]) if "amount" in df.columns else 0.0,
                "details": str(r.get("details", "")),
            })

    return pd.DataFrame(rows)


def _apply_universe_filter(codes: List[str], run_dates: List[str]) -> List[str]:
    """用快照对股票列表应用 universe 过滤

    快照缺失时返回原列表（数据不足不阻断信号生成）。
    """
    # 候选池过滤：按快照应用 universe 过滤，剔除不合格股票。
    # 回测窗口首日可能早于快照覆盖范围（快照从某个日期才开始生成），
    # 所以要取窗口内最近可用的快照，而不是用窗口首日。
    snap_dir = get_config("paths.snapshot", "data/snapshot")
    snap = _load_snapshot_near(snap_dir, run_dates)
    if snap is None:
        return codes

    if snap.empty or "code" not in snap.columns:
        return codes

    from src.universe.filter import filter_universe
    pool, _ = filter_universe(snap)
    pool_codes = set(pool["code"].astype(str).tolist())
    kept = [c for c in codes if c in pool_codes]
    if len(kept) < len(codes):
        print(f"[信号引擎] universe 过滤：{len(codes)} → {len(kept)} 只")
    return kept


def _load_snapshot_near(snap_dir: str, run_dates: List[str]) -> Optional[pd.DataFrame]:
    """在 run_dates 窗口内取最近可用的快照，找不到返回 None"""
    sdir = Path(snap_dir)
    if not sdir.exists():
        return None

    available = {p.stem for p in sdir.glob("*.parquet")}
    # 窗口内可用的日期
    in_window = sorted(d for d in run_dates if d in available)
    if in_window:
        date = in_window[0]
    else:
        # 窗口外最近的快照也用（数据缺失时不阻断）
        all_dates = sorted(available)
        date = all_dates[0] if all_dates else None
        if date:
            print(f"[信号引擎] 窗口内无快照，改用最近快照 {date}")

    if date is None:
        return None

    path = sdir / f"{date}.parquet"
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def generate_signals(run_dates: List[str], cfg_path: str = "config.yaml",
                     codes: List[str] = None) -> pd.DataFrame:
    """
    全市场扫描，生成信号表

    Returns:
        DataFrame 包含列: date/code/detector/score/close/pct_chg/amount/details
    """
    cfg = load_config(cfg_path)
    daily_dir = get_config("paths.daily", "data/daily")

    if codes is None:
        codes = list_codes(daily_dir)
        if not codes:
            print("[信号引擎] 无可用日线数据，请先运行 generate_data.py")
            return pd.DataFrame()

    # 候选池过滤：按快照应用 universe 过滤，剔除不合格股票
    # 快照缺失时跳过过滤（数据不足时不阻断信号生成）
    codes = _apply_universe_filter(codes, run_dates)

    detectors = build_detectors(cfg)
    scorer = WeightedScorer(cfg.get("scoring", {}).get("weights", {}))

    print(f"[信号引擎] 扫描 {len(codes)} 只股票 × {len(run_dates)} 个交易日")

    all_rows = []
    for code in codes:
        df = scan_stock(code, detectors, scorer, run_dates, daily_dir)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("[信号引擎] 未发现任何信号")
        return pd.DataFrame()

    signals = pd.concat(all_rows, ignore_index=True)
    print(f"[信号引擎] 共产生 {len(signals)} 条信号")

    # 按日期输出
    out_dir = Path(get_config("paths.signals", "data/signals"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in run_dates:
        day_df = signals[signals["date"] == d]
        if not day_df.empty:
            day_df.to_parquet(out_dir / f"{d}.parquet", index=False)

    signals.to_parquet(out_dir / "all_signals.parquet", index=False)
    return signals
