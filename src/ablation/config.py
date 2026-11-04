"""子模块消融回测配置。

在 config.yaml 的 `ablation` 段下，每项都是开关。
总策略代码不感知这些开关 —— 消融层通过 config override 注入，
所以子模块贡献可以独立测量，而不用改策略实现。
"""

ABLATION_DEFAULTS = {
    "universe": {
        # 逐条过滤条件开关：关掉某条，看它筛掉了什么、对信号质量的影响
        "exclude_st": True,
        "price_range": True,
        "market_cap_range": True,
        "pct_chg_limit": True,
        "amount_top_n": True,
        "overheat_exclude": True,
    },
    "scoring": {
        # 打分权重归零 = 该因子不参与选股
        "weights": {
            "pattern_fit": 30.0,
            "ma_trend": 20.0,
            "vol_health": 15.0,
            "relative_strength": 15.0,
            "pullback_position": 10.0,
            "macd": 10.0,
        },
        "buy_max_score": 70.0,
    },
    "risk": {
        # 逐条卖出规则开关。全部关闭 = 纯持有到期，是风控贡献的上限基准
        "R1_stop_loss": True,
        "R2_breakeven": True,
        "R3_trailing": True,
        "R4_hard_profit": True,
        "B1_ma_exit": True,
        "B2_time_stop": True,
        "B3_limit_down": True,
    },
    "sentiment": {
        "enabled": True,
        "gate_open": 52.0,
        "gate_limit": 40.0,
    },
    "position": {
        "max_buy_per_round": 2,
        "max_buy_per_day": 3,
        "max_holdings": 6,
        "target_weight": 0.18,
    },
    "buy_guards": {
        "no_chase_3pct": True,      # 现价 > 选股价 × 1.03 不追
        "no_limit_up": True,        # 涨停不买
        "cooldown_days": 30,        # 卖出后 N 天不买回
        "pool_stale_halt": True,    # 股票池落后即停买（F3）
    },
}


def merge_ablation(base: dict, override: dict) -> dict:
    """把 override 深合并到 base 的 ablation 段"""
    ab = dict(base.get("ablation", {}))
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(ab.get(k), dict):
            ab[k] = {**ab[k], **v}
        else:
            ab[k] = v
    return ab
