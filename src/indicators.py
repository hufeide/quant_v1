"""技术指标计算

所有指标均带 lookahead 防护：第 i 行的结果只使用 <= i 的数据，
避免回测中未来函数污染。
"""
import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    """简单移动平均，前 n-1 个值为 NaN"""
    return series.rolling(window=n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 指标

    返回 (dif, dea, hist)：
      dif  = EMA(fast) - EMA(slow)
      dea  = EMA(dif, signal)
      hist = 2 * (dif - dea)
    """
    ef = ema(close, fast)
    es = ema(close, slow)
    dif = ef - es
    dea = ema(dif, signal)
    hist = 2.0 * (dif - dea)
    return dif, dea, hist


def max_high(df: pd.DataFrame, n: int, col: str = "high") -> pd.Series:
    """滚动 N 日最高价（不含当日，用于"创新高"判断的前高基准）"""
    return df[col].rolling(window=n, min_periods=n).max().shift(1)


def min_low(df: pd.DataFrame, n: int, col: str = "low") -> pd.Series:
    """滚动 N 日最低价（不含当日）"""
    return df[col].rolling(window=n, min_periods=n).min().shift(1)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """为日线 DataFrame 批量添加全部衍生指标

    输入需包含列：open/high/low/close/volume/amount/pct_chg/turnover_rate/
                  market_cap_float/is_st/pre_close/limit_up/limit_down
    """
    out = df.copy()
    close = out["close"]

    # 均线族
    for n in (5, 10, 20, 60):
        out[f"ma{n}"] = sma(close, n)

    # MACD
    out["dif"], out["dea"], out["hist"] = macd(close)

    # 前高前低（不含当日，避免当日新高被自己比掉）
    out["high_60_prev"] = max_high(out, 60)
    out["low_10_prev"] = min_low(out, 10)
    out["high_20"] = out["high"].rolling(20, min_periods=20).max()
    out["high_10"] = out["high"].rolling(10, min_periods=10).max()

    # 均量（shift(1)：不含当日，否则量比恒被当日自身成交量拉向 1.0）
    out["vol_ma5"] = sma(out["volume"], 5).shift(1)
    out["vol_ma10"] = sma(out["volume"], 10).shift(1)
    out["vol_ma20"] = sma(out["volume"], 20).shift(1)

    # 区间涨幅
    out["ret_20"] = close / out["close"].shift(20) - 1.0
    out["ret_10"] = close / out["close"].shift(10) - 1.0

    # 距前高回撤
    out["dist_from_high_20"] = close / out["high_20"] - 1.0

    # 量比：当日量 / 5日均量
    out["vol_ratio_5"] = out["volume"] / out["vol_ma5"]

    # 成交额 10 日均额，用于情绪代理
    out["amount_ma10"] = sma(out["amount"], 10)
    out["amount_ratio"] = out["amount"] / out["amount_ma10"]

    return out


def detect_uptrend_and_pullback(df: pd.DataFrame, i: int, cfg: dict) -> dict:
    """识别"上涨波段 + 随后回调"结构，返回诊断信息

    cfg 参数：
      min_prior_gain: 上涨波段最小涨幅（%）
      min_pullback_days / max_pullback_days: 回调天数区间
      vol_shrink_ratio: 缩量比例
      ma_support: 支撑均线
    """
    if i is None or i < 60:
        return {"ok": False, "reason": "数据不足"}

    close = df["close"].values
    vol = df["volume"].values
    high = df["high"].values

    cfg_min_gain = cfg.get("min_prior_gain", 8.0)
    cfg_min_pb = cfg.get("min_pullback_days", 2)
    cfg_max_pb = cfg.get("max_pullback_days", 8)
    cfg_shrink = cfg.get("vol_shrink_ratio", 0.75)
    cfg_ma = cfg.get("ma_support", 20)

    ma_col = f"ma{cfg_ma}"
    if ma_col not in df.columns:
        return {"ok": False, "reason": f"缺少 {ma_col}"}
    ma_vals = df[ma_col].values

    # 枚举上涨波段终点 j 与回调起点 j+1，要求 i 落在回调区间内
    best = None
    for j in range(max(20, i - cfg_max_pb - 40), i - cfg_min_pb + 1):
        if j - 20 < 0:
            continue
        seg_ret = (close[j] / close[j - 20] - 1.0) * 100.0
        if seg_ret < cfg_min_gain:
            continue
        pb_days = i - j
        if not (cfg_min_pb <= pb_days <= cfg_max_pb):
            continue
        # 回调期跌幅（相对波段高点）
        dd = (close[i] / high[j] - 1.0) * 100.0
        # 回调期均量 vs 上涨期均量
        up_vol = float(np.mean(vol[j - 20:j]))
        pb_vol = float(np.mean(vol[j + 1:i + 1])) if j + 1 <= i else 0.0
        if up_vol <= 0:
            continue
        vol_ratio = pb_vol / up_vol
        # MA 支撑必须逐日比较：回调期每一天的 low 与当天的 MA 比较。
        # 用 min(low) 对末日 MA 比较不等价——MA 在回调期是变动的，
        # 那样会漏掉回调中段跌破均线的情况（实测约 9% 的场景）。
        lows_pb = df["low"].values[j + 1:i + 1]
        mas_pb = ma_vals[j + 1:i + 1]
        if np.any(np.isnan(mas_pb)) or len(lows_pb) != len(mas_pb):
            continue
        ma_ok = bool(np.all(lows_pb >= mas_pb))
        score = seg_ret * 0.5 + (100.0 if ma_ok else 0.0) - abs(dd) * 0.1
        if best is None or score > best["score"]:
            best = {
                "score": score, "ok": ma_ok, "reason": "" if ma_ok else "跌破支撑均线",
                "seg_ret": seg_ret, "pb_days": pb_days, "drawdown": dd,
                "vol_ratio": vol_ratio, "band_end": int(j),
            }
    return best if best else {"ok": False, "reason": "未找到合格上涨-回调结构"}
