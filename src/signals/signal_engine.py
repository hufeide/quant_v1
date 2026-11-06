"""信号引擎：日线扫描 → 形态检测 → 综合打分 → 原始信号输出

架构位置：信号层（signal layer）。
流程：
  1. 加载全市场日线数据
  2. 逐日时点候选池粗筛（universe filter，point-in-time 无前视）
  3. 对每只候选股扫描全部交易日，运行检测器
  4. 计算综合打分，输出原始信号

本模块只产出"原始信号"，不做有效性过滤、不做排序（两者依次在
signal_filter / signal_ranker 完成），不决定买多少股、不下单。

原始信号行结构（示例）：
    {date, code, detector, signal="buy", entry_price, atr,
     score, pct_chg, amount, details}
"""
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from src.config import load_config, get_config, module_enabled
from src.data_loader import load_daily_with_indicators, list_codes
from src.indicators import add_indicators
from src.signals.scoring.weighted_scorer import WeightedScorer
from src.signals.detectors.pullback import PullbackDetector
from src.signals.detectors.breakout import BreakoutDetector
from src.signals.detectors.oversold import OversoldDetector
from src.signals.detectors.momentum import MomentumDetector


def build_detectors(cfg: Dict) -> Dict:
    """构建检测器实例

    开关（见 config.yaml modules.detectors）：
      - enabled：总开关，False 时不产生任何检测信号
      - modules.detectors.<name>：各检测器独立开关
      - 兼容旧位置 detectors.<name>.enabled（任一为 False 即停用）
    """
    if not module_enabled("detectors.enabled"):
        return {}

    dc = cfg.get("detectors", {})
    out = {}
    for name, cls in (("pullback", PullbackDetector), ("breakout", BreakoutDetector),
                      ("oversold", OversoldDetector), ("momentum", MomentumDetector)):
        on_modules = module_enabled(f"detectors.{name}", True)
        on_legacy = (dc.get(name) or {}).get("enabled", True)
        if on_modules and on_legacy:
            out[name] = cls(dc.get(name, {}))
    return out


def scan_stock(code: str, detectors: Dict, scorer: WeightedScorer,
               run_dates: List[str], daily_dir: str,
               rs_map: Dict[tuple, float] = None,
               universe_by_date: Dict[str, Optional[set]] = None) -> pd.DataFrame:
    """对单只股票扫描指定日期范围，输出原始信号行

    rs_map: 预先算好的横截面相对强度百分位 {(date, code): percentile}。
        未传或缺失时回退 0.5（中性），不阻断扫描。
    universe_by_date: 逐日 point-in-time 候选池 {date: set(codes)}，
        date 当日不在池中的股票不产生该日信号；值为 None 的日期不过滤。

    打分阈值（buy_max_score）过滤不在这里做——属于 signal_filter 的职责。
    """
    rs_map = rs_map or {}
    universe_by_date = universe_by_date or {}
    try:
        df = load_daily_with_indicators(code, daily_dir)
    except FileNotFoundError:
        return pd.DataFrame()

    if "date" not in df.columns or df.empty:
        return pd.DataFrame()

    rows = []
    date_set = set(run_dates)
    n = len(df)

    # 预计算：仅保留本股在当前 universe 中可行的交易日子集
    # （universe_by_date 已保证只含当日及之前的数据，无前视；
    #   值为 None 的日期无快照可用，按"不过滤"处理）
    eligible_dates = (
        {d for d in date_set
         if d in universe_by_date
         and (universe_by_date[d] is None or code in universe_by_date[d])}
        if universe_by_date else set(date_set)
    )
    if universe_by_date and not eligible_dates:
        return pd.DataFrame()

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
        if universe_by_date and date_str not in eligible_dates:
            continue  # 当日未入池：不进过滤（前视偏差防护）

        for dname, (det, enabled) in det_results.items():
            if date_str not in enabled:
                continue
            try:
                r = det.detect(df, i)
            except Exception as e:
                # 记录异常但不阻断扫描（某只股票数据异常不应影响其他股票）
                import logging
                logging.debug(f"[scan_stock] {code} date={date_str} detector={dname} error: {e}")
                continue

            if not r.get("detected"):
                continue

            # 相对强度：当日全市场 ret_20 横截面百分位 [0,1]
            # 由引擎层预算（见 compute_relative_strength），缺失时中性 0.5
            relative_rank = rs_map.get((date_str, code), 0.5)

            # 评分器开关（modules.scoring.enabled）：关闭时不打分，
            # score 记为检测器自带的形态契合度分。
            # 注意：阈值过滤（buy_max_score）已移交 signal_filter。
            if module_enabled("scoring.enabled"):
                score = scorer.score(df, i, r, relative_rank)
            else:
                score = float(r.get("score", 0.0))

            if "atr14" in df.columns:
                atr_val = df["atr14"].iloc[i]
                atr_val = float(atr_val) if pd.notna(atr_val) else 0.0
            else:
                atr_val = 0.0

            close = float(df["close"].iloc[i])

            rows.append({
                "date": date_str,
                "code": code,
                "detector": dname,
                "signal": "buy",
                "entry_price": close,
                "atr": atr_val,
                "score": score,
                "pct_chg": float(df["pct_chg"].iloc[i]) if "pct_chg" in df.columns else 0.0,
                "amount": float(df["amount"].iloc[i]) if "amount" in df.columns else 0.0,
                "details": str(r.get("details", "")),
            })

    return pd.DataFrame(rows)


def apply_pipeline(raw: pd.DataFrame, *, universe_by_date=None,
                   order_by: str = "score") -> pd.DataFrame:
    """原始信号 -> 有效信号 -> 排序结果（完整信号流水线）

    signal_engine 组合 signal_filter + signal_ranker：
      - filter_signals: 检测器开关、打分阈值、universe 成员资格、字段有效性
      - rank_signals:   按评分等维度排序

    排序仍由 signal_ranker 完成；这里只传参。
    """
    from src.signals.signal_filter import filter_signals
    from src.signals.signal_ranker import rank_signals

    valid = filter_signals(raw, universe_by_date=universe_by_date)
    return rank_signals(valid, order_by=order_by)


def _build_universe_by_date(run_dates: List[str]) -> Dict[str, Optional[set]]:
    """为每个交易日构建 point-in-time 候选池

    对每个交易日，只取该日 **之前（含当日）最近** 的快照应用 universe 过滤。
    这样某日之后才进池的股票不会在更早的交易日产生信号，避免前视偏差。
    快照文件缺失/读取失败记为 None（该日不过滤，数据不足不阻断信号生成）。

    Returns:
        {date: set(candidate codes) | None}
    """
    snap_dir = get_config("paths.snapshot", "data/snapshot")
    sdir = Path(snap_dir)
    if not sdir.exists():
        return {d: None for d in run_dates}

    available = sorted(p.stem for p in sdir.glob("*.parquet"))
    if not available:
        return {d: None for d in run_dates}

    from src.universe.filter import filter_universe

    pool_cache: Dict[str, Optional[set]] = {}
    out: Dict[str, Optional[set]] = {}
    for d in run_dates:
        # 在该日及之前最近的快照（bisect_right - 1），而不是窗口内最晚的
        idx = bisect_right(available, d) - 1
        if idx < 0:
            # 早于最早可用快照：该日无历史 universe 信息，不过滤
            out[d] = None
            continue
        snap_date = available[idx]
        if snap_date not in pool_cache:
            try:
                snap = pd.read_parquet(sdir / f"{snap_date}.parquet")
                if snap.empty or "code" not in snap.columns:
                    pool_cache[snap_date] = None
                else:
                    pool, _ = filter_universe(snap)
                    pool_cache[snap_date] = set(pool["code"].astype(str))
            except Exception:
                pool_cache[snap_date] = None
        out[d] = pool_cache[snap_date]
    return out


def compute_relative_strength(
    codes: List[str], run_dates: List[str], daily_dir: str,
    ret_col: str = "ret_20",
) -> Dict[tuple, float]:
    """预算横截面相对强度百分位

    对每个交易日，把全市场 ret_col（如 ret_20）排序，输出每只股票的
    百分位 [0,1]。1.0 = 当日全市场最强，0.0 = 最弱。

    只在 run_dates 窗口内计算，避免为预热期浪费内存。
    单次全市场扫描（约 5000 只 × 90 日 ≈ 45 万行），用向量化 rank 计算。
    """
    date_set = set(run_dates)
    # 先收集所有 (date, code, ret) 三元组
    all_rets = []   # list of (date, code, ret)
    for code in codes:
        try:
            df = load_daily_with_indicators(code, daily_dir)
        except FileNotFoundError:
            continue
        if df.empty or ret_col not in df.columns:
            continue
        for i in range(len(df)):
            d = str(df["date"].iloc[i])[:10]
            if d not in date_set:
                continue
            r = df[ret_col].iloc[i]
            if pd.isna(r):
                continue
            all_rets.append((d, code, float(r)))

    if not all_rets:
        return {}

    # 转 DataFrame 并按日分组排序
    raw = pd.DataFrame(all_rets, columns=["date", "code", "ret"])
    out: Dict[tuple, float] = {}
    for d, g in raw.groupby("date", sort=False):
        # 升序 rank 后除以总数 → 百分位（最大值 → 1.0）
        ranks = g["ret"].rank(method="average")
        n = len(g)
        if n < 2:
            # 只有一只股票时百分位无意义，给 0.5
            for code in g["code"]:
                out[(d, code)] = 0.5
            continue
        for code, rank in zip(g["code"], ranks):
            out[(d, code)] = round(float(rank) / n, 4)

    return out


def generate_signals(run_dates: List[str], cfg_path: str = "config.yaml",
                     codes: List[str] = None, cfg: dict = None) -> pd.DataFrame:
    """全市场扫描 → 原始信号 → 过滤 → 排序 → 写入信号表

    参数 cfg 可传入一个已完成变体注入的配置 dict
        （不走全局 load_config + override），用于消融实验隔离信号输出目录。

    Returns:
        DataFrame 包含列:
        date/code/detector/signal/entry_price/atr/score/pct_chg/amount/details
    """
    cfg = cfg if cfg is not None else load_config(cfg_path)
    daily_dir = get_config("paths.daily", "data/daily")

    if codes is None:
        codes = list_codes(daily_dir)
        if not codes:
            print("[信号引擎] 无可用日线数据，请先运行 generate_data.py")
            return pd.DataFrame()

    # 候选池过滤：按交易日构建 point-in-time universe
    # （每日只用当日及之前最近快照，无前视；详见 _build_universe_by_date）
    universe_by_date = _build_universe_by_date(run_dates)

    # 粗筛：只保留任一日期入池过的股票（universe 使用了并集上界，不会再遗漏候选）
    n_with_snap = sum(1 for u in universe_by_date.values() if u is not None)
    if n_with_snap:
        union_pool = set().union(*[u for u in universe_by_date.values() if u])
        if union_pool:
            codes = [c for c in codes if c in union_pool]
        print(f"[信号引擎] universe 按日过滤：{len(universe_by_date)} 天，"
            f"{n_with_snap} 天有快照")

    detectors = build_detectors(cfg)
    scorer = WeightedScorer(cfg.get("scoring", {}).get("weights", {}))

    print(f"[信号引擎] 扫描 {len(codes)} 只股票 × {len(run_dates)} 个交易日")

    # 相对强度百分位预算
    rs_map = compute_relative_strength(codes, run_dates, daily_dir)
    print(f"[信号引擎] 相对强度百分位已计算 {len(rs_map)} 条")

    all_rows = []
    for code in codes:
        df = scan_stock(code, detectors, scorer, run_dates, daily_dir,
                         rs_map=rs_map, universe_by_date=universe_by_date)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("[信号引擎] 未发现任何信号")
        return pd.DataFrame()

    raw_signals = pd.concat(all_rows, ignore_index=True)
    print(f"[信号引擎] 共产生 {len(raw_signals)} 条原始信号")

    # 过滤 → 排序：原始信号进流水线
    signals = apply_pipeline(
        raw_signals,
        universe_by_date=universe_by_date,
        order_by=cfg.get("ablation", {}).get("order", {}).get("by", "score"),
    )
    print(f"[信号引擎] 过滤后有效信号 {len(signals)} 条")

    # 按日期输出。out_dir 优先取传入 cfg 的 paths.signals
    # （消融实验需要隔离变体信号，不覆盖共享的 data/signals）
    out_dir = Path(cfg.get("paths", {}).get("signals", get_config("paths.signals", "data/signals")))
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in run_dates:
        day_df = signals[signals["date"] == d]
        if not day_df.empty:
            day_df.to_parquet(out_dir / f"{d}.parquet", index=False)

    signals.to_parquet(out_dir / "all_signals.parquet", index=False)
    return signals
