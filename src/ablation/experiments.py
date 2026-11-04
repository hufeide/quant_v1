"""消融实验定义：每个子模块的独立开关实验。

按模块组织，每项实验：
  - 只改该模块的配置
  - 其他模块保持基线
  - 返回修改后的配置
"""
from copy import deepcopy
from typing import Callable, Dict
from src.ablation.config import ABLATION_DEFAULTS


def baseline_cfg() -> dict:
    """返回完整的消融配置基线"""
    return deepcopy(ABLATION_DEFAULTS)


# ==================== Universe 过滤 ====================

def universe_exclude_st_off(cfg: dict) -> dict:
    """不剔 ST 股"""
    c = deepcopy(cfg)
    c["universe"]["exclude_st"] = False
    return c


def universe_no_price_range(cfg: dict) -> dict:
    """不限制股价区间"""
    c = deepcopy(cfg)
    c["universe"]["price_range"] = False
    return c


def universe_no_market_cap(cfg: dict) -> dict:
    """不限制流通市值"""
    c = deepcopy(cfg)
    c["universe"]["market_cap_range"] = False
    return c


def universe_no_pct_chg_limit(cfg: dict) -> dict:
    """不限制涨跌幅（追涨杀跌）"""
    c = deepcopy(cfg)
    c["universe"]["pct_chg_limit"] = False
    return c


def universe_no_amount_top_n(cfg: dict) -> dict:
    """不按成交额排序（失去流动性筛选）"""
    c = deepcopy(cfg)
    c["universe"]["amount_top_n"] = False
    return c


def universe_no_overheat(cfg: dict) -> dict:
    """不剔除换手率过热股"""
    c = deepcopy(cfg)
    c["universe"]["overheat_exclude"] = False
    return c


# ==================== Scoring 打分 ====================

def scoring_zero_pattern_fit(cfg: dict) -> dict:
    """打分去掉形态契合度因子"""
    c = deepcopy(cfg)
    w = c["scoring"]["weights"]
    w["pattern_fit"] = 0.0
    # 归一化其他权重
    total = sum(w.values())
    if total > 0:
        for k in w:
            w[k] = w[k] / total * 100.0
    return c


def scoring_zero_ma_trend(cfg: dict) -> dict:
    """打分去掉均线趋势因子"""
    c = deepcopy(cfg)
    w = c["scoring"]["weights"]
    w["ma_trend"] = 0.0
    total = sum(w.values())
    if total > 0:
        for k in w:
            w[k] = w[k] / total * 100.0
    return c


def scoring_zero_vol_health(cfg: dict) -> dict:
    """打分去掉缩量健康度因子"""
    c = deepcopy(cfg)
    w = c["scoring"]["weights"]
    w["vol_health"] = 0.0
    total = sum(w.values())
    if total > 0:
        for k in w:
            w[k] = w[k] / total * 100.0
    return c


def scoring_zero_relative_strength(cfg: dict) -> dict:
    """打分去掉相对强度因子"""
    c = deepcopy(cfg)
    w = c["scoring"]["weights"]
    w["relative_strength"] = 0.0
    total = sum(w.values())
    if total > 0:
        for k in w:
            w[k] = w[k] / total * 100.0
    return c


def scoring_zero_pullback_position(cfg: dict) -> dict:
    """打分去掉回踩位置因子"""
    c = deepcopy(cfg)
    w = c["scoring"]["weights"]
    w["pullback_position"] = 0.0
    total = sum(w.values())
    if total > 0:
        for k in w:
            w[k] = w[k] / total * 100.0
    return c


def scoring_zero_macd(cfg: dict) -> dict:
    """打分去掉 MACD 因子"""
    c = deepcopy(cfg)
    w = c["scoring"]["weights"]
    w["macd"] = 0.0
    total = sum(w.values())
    if total > 0:
        for k in w:
            w[k] = w[k] / total * 100.0
    return c


# ==================== Risk 风控 ====================

def risk_R1_off(cfg: dict) -> dict:
    """关闭 R1 止损"""
    c = deepcopy(cfg)
    c["risk"]["R1_stop_loss"] = False
    return c


def risk_R2_off(cfg: dict) -> dict:
    """关闭 R2 保盈止损"""
    c = deepcopy(cfg)
    c["risk"]["R2_breakeven"] = False
    return c


def risk_R3_off(cfg: dict) -> dict:
    """关闭 R3 移动止盈"""
    c = deepcopy(cfg)
    c["risk"]["R3_trailing"] = False
    return c


def risk_R4_off(cfg: dict) -> dict:
    """关闭 R4 硬止盈"""
    c = deepcopy(cfg)
    c["risk"]["R4_hard_profit"] = False
    return c


def risk_B1_off(cfg: dict) -> dict:
    """关闭 B1 跌破均线收盘确认"""
    c = deepcopy(cfg)
    c["risk"]["B1_ma_exit"] = False
    return c


def risk_B2_off(cfg: dict) -> dict:
    """关闭 B2 时间止损"""
    c = deepcopy(cfg)
    c["risk"]["B2_time_stop"] = False
    return c


def risk_all_off(cfg: dict) -> dict:
    """关闭全部风控（纯持有到期）"""
    c = deepcopy(cfg)
    for k in c["risk"]:
        if k.endswith("_off") or k.endswith("_stop_loss") or k.endswith("_breakeven") or k.endswith("_trailing") or k.endswith("_profit") or k.endswith("_ma_exit") or k.endswith("_time_stop"):
            c["risk"][k] = False
    return c


# ==================== Sentiment 闸门 ====================

def sentiment_disabled(cfg: dict) -> dict:
    """关闭情绪闸门（不受市场情绪影响）"""
    c = deepcopy(cfg)
    c["sentiment"]["enabled"] = False
    return c


def sentiment_relaxed(cfg: dict) -> dict:
    """放宽闸门：开放阈值降到 30"""
    c = deepcopy(cfg)
    c["sentiment"]["gate_open"] = 30.0
    c["sentiment"]["gate_limit"] = 15.0
    return c


def sentiment_strict(cfg: dict) -> dict:
    """收紧闸门：开放阈值升到 70"""
    c = deepcopy(cfg)
    c["sentiment"]["gate_open"] = 70.0
    c["sentiment"]["gate_limit"] = 55.0
    return c


# ==================== Position 仓位 ====================

def position_unlimited_per_round(cfg: dict) -> dict:
    """每轮不限买入数量"""
    c = deepcopy(cfg)
    c["position"]["max_buy_per_round"] = 999
    return c


def position_max_1_per_round(cfg: dict) -> dict:
    """每轮最多买 1 只"""
    c = deepcopy(cfg)
    c["position"]["max_buy_per_round"] = 1
    return c


def position_unlimited_holdings(cfg: dict) -> dict:
    """不限持仓只数"""
    c = deepcopy(cfg)
    c["position"]["max_holdings"] = 999
    return c


def position_equal_weight(cfg: dict) -> dict:
    """等权重建仓（5% 每只）"""
    c = deepcopy(cfg)
    c["position"]["target_weight"] = 0.05
    return c


# ==================== Buy Guards 买入保护 ====================

def buy_no_chase_guard(cfg: dict) -> dict:
    """允许追高 3%"""
    c = deepcopy(cfg)
    c["buy_guards"]["no_chase_3pct"] = False
    return c


def buy_allow_limit_up(cfg: dict) -> dict:
    """允许买涨停股"""
    c = deepcopy(cfg)
    c["buy_guards"]["no_limit_up"] = False
    return c


def buy_no_cooldown(cfg: dict) -> dict:
    """卖出后立即可买回（无冷却期）"""
    c = deepcopy(cfg)
    c["buy_guards"]["cooldown_days"] = 0
    return c


def buy_no_pool_stale_halt(cfg: dict) -> dict:
    """股票池落后时仍允许买入"""
    c = deepcopy(cfg)
    c["buy_guards"]["pool_stale_halt"] = False
    return c


# ==================== 实验注册表 ====================

ABLATION_EXPERIMENTS = {
    # Universe
    "universe_no_exclude_st": (
        "不剔 ST 股",
        "universe",
        universe_exclude_st_off,
    ),
    "universe_no_price_range": (
        "不限制股价区间",
        "universe",
        universe_no_price_range,
    ),
    "universe_no_market_cap": (
        "不限制流通市值",
        "universe",
        universe_no_market_cap,
    ),
    "universe_no_pct_chg": (
        "不限制涨跌幅",
        "universe",
        universe_no_pct_chg_limit,
    ),
    "universe_no_amount_top": (
        "不按成交额排序",
        "universe",
        universe_no_amount_top_n,
    ),
    "universe_no_overheat": (
        "不剔除换手率过热",
        "universe",
        universe_no_overheat,
    ),
    # Scoring
    "scoring_zero_pattern_fit": (
        "打分去掉形态契合度",
        "scoring",
        scoring_zero_pattern_fit,
    ),
    "scoring_zero_ma_trend": (
        "打分去掉均线趋势",
        "scoring",
        scoring_zero_ma_trend,
    ),
    "scoring_zero_vol_health": (
        "打分去掉缩量健康度",
        "scoring",
        scoring_zero_vol_health,
    ),
    "scoring_zero_relative_strength": (
        "打分去掉相对强度",
        "scoring",
        scoring_zero_relative_strength,
    ),
    "scoring_zero_pullback_position": (
        "打分去掉回踩位置",
        "scoring",
        scoring_zero_pullback_position,
    ),
    "scoring_zero_macd": (
        "打分去掉 MACD",
        "scoring",
        scoring_zero_macd,
    ),
    # Risk
    "risk_R1_off": ("关闭 R1 止损", "risk", risk_R1_off),
    "risk_R2_off": ("关闭 R2 保盈", "risk", risk_R2_off),
    "risk_R3_off": ("关闭 R3 移动止盈", "risk", risk_R3_off),
    "risk_R4_off": ("关闭 R4 硬止盈", "risk", risk_R4_off),
    "risk_B1_off": ("关闭 B1 跌破均线", "risk", risk_B1_off),
    "risk_B2_off": ("关闭 B2 时间止损", "risk", risk_B2_off),
    "risk_all_off": ("关闭全部风控", "risk", risk_all_off),
    # Sentiment
    "sentiment_disabled": ("关闭情绪闸门", "sentiment", sentiment_disabled),
    "sentiment_relaxed": ("放宽闸门到 30", "sentiment", sentiment_relaxed),
    "sentiment_strict": ("收紧闸门到 70", "sentiment", sentiment_strict),
    # Position
    "position_unlimited_per_round": ("每轮不限买入数", "position", position_unlimited_per_round),
    "position_max_1_per_round": ("每轮最多买 1 只", "position", position_max_1_per_round),
    "position_unlimited_holdings": ("不限持仓只数", "position", position_unlimited_holdings),
    "position_equal_weight": ("等权重 5% 建仓", "position", position_equal_weight),
    # Buy Guards
    "buy_no_chase_guard": ("允许追高 3%", "buy_guards", buy_no_chase_guard),
    "buy_allow_limit_up": ("允许买涨停", "buy_guards", buy_allow_limit_up),
    "buy_no_cooldown": ("卖出后无冷却", "buy_guards", buy_no_cooldown),
    "buy_no_pool_stale_halt": ("股票池落后仍可买", "buy_guards", buy_no_pool_stale_halt),
}


def get_experiment(name: str):
    """按名称取实验定义"""
    if name not in ABLATION_EXPERIMENTS:
        raise KeyError(f"未知实验: {name}, 可用: {list(ABLATION_EXPERIMENTS.keys())}")
    desc, module, toggle_fn = ABLATION_EXPERIMENTS[name]
    return {"description": desc, "module": module, "toggle_fn": toggle_fn}
