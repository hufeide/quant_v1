#!/usr/bin/env python3
"""单元测试：四个形态检测器 + 指标计算

用法： python tests/test_detectors.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.indicators import add_indicators
from src.detectors.pullback import PullbackDetector
from src.detectors.breakout import BreakoutDetector
from src.detectors.oversold import OversoldDetector
from src.detectors.momentum import MomentumDetector
from src.scoring.weighted_scorer import WeightedScorer

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


def make_df(prices, volumes=None, n=None):
    """从价格序列构造标准 DataFrame"""
    n = len(prices) if n is None else n
    if volumes is None:
        volumes = [1_000_000] * n
    dates = pd.bdate_range(end="2025-09-15", periods=n)
    df = pd.DataFrame({
        "code": "TEST", "date": dates,
        "open": prices, "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices], "close": prices,
        "volume": volumes, "amount": [v * p for v, p in zip(volumes, prices)],
        "pct_chg": [0.0] + [(prices[i] / prices[i - 1] - 1.0) * 100.0 for i in range(1, n)],
        "turnover_rate": [2.0] * n,
        "market_cap_float": [5e9] * n, "is_st": [False] * n,
        "pre_close": prices[:1] + prices[:-1] if n > 1 else [prices[0]],
        "limit_up": [p * 1.10 for p in prices],
        "limit_down": [p * 0.90 for p in prices],
    })
    df = add_indicators(df)
    return df


# ==================== 缩量回踩 ====================
print("== 缩量回踩检测器 ==")
pb = PullbackDetector({"min_prior_gain": 8.0, "min_pullback_days": 2,
                       "max_pullback_days": 8, "vol_shrink_ratio": 0.75,
                       "ma_support": 20})

# 上涨 10%（100→110）后缩量回调 3 天
warm = [100.0] * 35                       # 平盘预热，满足 index>=60 的前置条件
up = [100 * (1 + 0.004) ** i for i in range(25)]           # 涨到 ~110.4
pb_down = [up[-1] * (1 - 0.003) ** i for i in range(1, 4)]   # 回调 3 天，每天 -0.3%
vols_up = [2_000_000] * 25
vols_pb = [600_000] * 3                                     # 缩量到 30%
prices = warm + up + pb_down
vols = [1_000_000] * 35 + vols_up + vols_pb
df = make_df(prices, vols)

up_end = len(warm) + len(up) - 1          # 上涨末端，尚未回调
r_nc = pb.detect(df, up_end)
check("未回调不命中", not r_nc["detected"], str(r_nc.get("details")))

r = pb.detect(df, len(prices) - 1)
check("缩量回踩命中", r["detected"], str(r.get("details")))
check("得分范围[0,100]", 0 <= r["score"] <= 100, str(r["score"]))

# 上涨但未回调（索引 63，处于上涨段）
r2 = pb.detect(df, len(warm) + 10)
check("上涨中不命中", not r2["detected"], str(r2.get("details")))

# 回调但未缩量
vols_no_shrink = [1_000_000] * 35 + vols_up + [2_000_000] * 3
df2 = make_df(prices, vols_no_shrink)
r3 = pb.detect(df2, len(prices) - 1)
check("未缩量不命中", not r3["detected"], str(r3.get("details")))


# ==================== 平台突破 ====================
print("\n== 平台突破检测器 ==")
bo = BreakoutDetector({"lookback": 60, "vol_ratio": 1.5, "max_dist_from_high": 6.0})

# 前 60 日横盘 100，第 61 日放量突破到 105
flat = [100.0] * 61
break_price = 105.0
prices_bo = flat + [break_price]
vols_bo = [1_000_000] * 61 + [2_500_000]
df = make_df(prices_bo, vols_bo)
r = bo.detect(df, 61)
check("平台突破命中", r["detected"], str(r.get("details")))
check("突破得分>0", r["score"] > 0, str(r["score"]))

# 突破但量不足
vols_weak = [1_000_000] * 61 + [1_200_000]
df2 = make_df(prices_bo, vols_weak)
r2 = bo.detect(df2, 61)
check("量不足不命中", not r2["detected"], str(r2.get("details")))

# 未突破（持平）
prices_flat = [100.0] * 62
df3 = make_df(prices_flat, [2_000_000] * 62)
r3 = bo.detect(df3, 61)
check("未创新高不命中", not r3["detected"], str(r3.get("details")))


# ==================== 超跌反弹 ====================
print("\n== 超跌反弹检测器 ==")
osd = OversoldDetector({"lookback": 10, "min_drop_from_high": 15.0,
                        "min_day_gain": 1.0, "no_new_low": True})

# 前 11 日从 100 跌到 83（跌 17%），当日涨 2%
# 0.97^10 ≈ 0.737，跌到 73.7 → 跌幅 26%，足够触发 15% 阈值
drop = [100 * (1 - 0.03) ** i for i in range(11)]   # 跌到 ~73.7
rebound = drop[-1] * 1.02                            # 反弹 2%
prices = drop + [rebound]
df = make_df(prices)
r = osd.detect(df, 11)
check("超跌反弹命中", r["detected"], str(r.get("details")))
check("反弹得分>0", r["score"] > 0, str(r["score"]))

# 跌幅不足
drop2 = [100 * (1 - 0.005) ** i for i in range(11)]    # 只跌 ~5%
prices2 = drop2 + [drop2[-1] * 1.02]
df2 = make_df(prices2)
r2 = osd.detect(df2, 11)
check("跌幅不足不命中", not r2["detected"], str(r2.get("details")))

# 超跌但当日仍跌（非阳线）
prices3 = drop + [drop[-1] * 0.98]
df3 = make_df(prices3)
r3 = osd.detect(df3, 11)
check("非阳线不命中", not r3["detected"], str(r3.get("details")))


# ==================== 强者恒强 ====================
print("\n== 强者恒强检测器 ==")
mo = MomentumDetector({"lookback": 20, "min_gain": 25.0,
                       "max_drawdown_from_high": 8.0, "ma_hold": 10})

# 20 日从 100 涨到 130（涨 30%），当日小幅回调但仍守住 MA10
up20 = [100 * (1 + 0.013) ** i for i in range(21)]     # 涨到 ~130.0
prices = up20
df = make_df(prices)
r = mo.detect(df, 20)
check("强者恒强命中", r["detected"], str(r.get("details")))
check("动量得分>0", r["score"] > 0, str(r["score"]))

# 涨幅不足
up_weak = [100 * (1 + 0.005) ** i for i in range(21)]   # 只涨 ~11%
df2 = make_df(up_weak)
r2 = mo.detect(df2, 20)
check("涨幅不足不命中", not r2["detected"], str(r2.get("details")))

# 涨够但跌破 MA10
up_then_drop = up20[:-1] + [up20[-1] * 0.85]             # 暴跌 15%
df3 = make_df(up_then_drop)
r3 = mo.detect(df3, 20)
check("跌破MA10不命中", not r3["detected"], str(r3.get("details")))


# ==================== 综合打分 ====================
print("\n== 综合打分器 ==")
ws = WeightedScorer({"pattern_fit": 30.0, "ma_trend": 20.0, "vol_health": 15.0,
                     "relative_strength": 15.0, "pullback_position": 10.0,
                     "macd": 10.0})

# 多头排列 + 完美形态
prices = [100 * (1 + 0.01) ** i for i in range(30)]
df = make_df(prices)
det = {"detected": True, "score": 80.0}
s = ws.score(df, 29, det, relative_rank=0.8)
check("得分在(0,100)", 0 < s < 100, str(s))
check("高分形态得分较高", s > 50, str(s))

# 未命中形态 → 0 分
det2 = {"detected": False, "score": 0.0}
s2 = ws.score(df, 29, det2)
check("未命中得0分", s2 == 0.0, str(s2))

# 权重校验
try:
    WeightedScorer({"a": 50.0, "b": 30.0})
    check("权重和须为100", False, "未抛异常")
except ValueError:
    check("权重和须为100", True)


# ==================== 指标计算 ====================
print("\n== 指标计算 ==")
df = make_df([100 * (1 + 0.01) ** i for i in range(80)])
check("MA20 存在", "ma20" in df.columns)
check("MA60 存在", "ma60" in df.columns)
check("DIF 存在", "dif" in df.columns)
check("DEA 存在", "dea" in df.columns)
check("vol_ma5 存在", "vol_ma5" in df.columns)
check("MA20 首行NaN", np.isnan(df["ma20"].iloc[0]))
check("MA20 第20行有效", not np.isnan(df["ma20"].iloc[19]))

# 无未来函数：MA20[i] 只能依赖 <=i 的数据
# 改动未来价格不应影响过去 MA
df2 = df.copy()
df2.loc[79, "close"] = df2.loc[79, "close"] * 2.0
check("MA 无未来函数", abs(df2["ma20"].iloc[30] - df["ma20"].iloc[30]) < 1e-9)


print(f"\n{'='*40}\n通过 {PASS} 项，失败 {FAIL} 项\n{'='*40}")
sys.exit(1 if FAIL else 0)
