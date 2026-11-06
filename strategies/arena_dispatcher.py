"""竞技场四策略在同一 Cerebro 中的分发策略

backtrader 的一个 Cerebro 只实例化一个 Strategy，
但竞技场需要 4 个独立账户。因此本分发器持有 4 个
(detector, Portfolio) 组合，把信号路由到对应账户。

共用约束：数据源、信号表、情绪指数、风控规则完全一致，
唯一变量是「买什么」（detector 决定）。这正是 V1 控制变量法的实现。
"""
from strategies.arena_base import ArenaStrategyBase
from src.config import get_config, module_enabled


class ArenaDispatcher(ArenaStrategyBase):
    """四账户分发器"""

    params = dict(
        detectors=None,        # {name: detector实例}
        account_ids=None,      # {name: account_id}
        detector_names=None,   # 需要运行的策略名列表
    )

    def __init__(self):
        # 基类会按 params 建一个占位 portfolio，此处不再重复
        # 基类的 next() 会为 self.portfolio 建一套账簿，竞技场用不到，
        # 但仍需要它初始化 data_map / ma_map / broker_sim / risk / gate，
        # 因此先调一次基类构造，再把四个账户挂上来。
        super().__init__()

        self.accounts = {}      # {detector_name: Portfolio}
        self.base_portfolio = self.portfolio

        for name in (self.p.detector_names or []):
            det = (self.p.detectors or {}).get(name)
            if det is None:
                continue
            aid = (self.p.account_ids or {}).get(name, name)
            pf = self._make_portfolio(aid)
            self.accounts[name] = pf

        # 统一风控/闸门（四账户共用同一套规则）
        self.risk = type(self.risk)(get_config("risk", {}))

    def _make_portfolio(self, account_id: str):
        """按账户号新建独立 Portfolio"""
        from src.portfolio import Portfolio
        return Portfolio(
            account_id=account_id,
            cash=self.p.initial_cash,
            cfg={
                "cooldown_days": get_config("cooldown.days", 30),
                "max_buy_per_round": get_config("position.max_buy_per_round", 2),
                "max_buy_per_day": get_config("position.max_buy_per_day", 3),
                "max_holdings": get_config("position.max_holdings", 6),
                "target_weight": get_config("position.target_weight", 0.18),
                "buy_max_score": get_config("scoring.buy_max_score", 70.0),
                # 组合管理开关（config.yaml modules.portfolio / modules.scoring）
                "portfolio_enabled": module_enabled("portfolio.enabled"),
                "holdings_limit": module_enabled("portfolio.holdings_limit"),
                "cooldown_enabled": module_enabled("portfolio.cooldown"),
                "nav_tracking": True,   # 净值追踪默认打开，不再受模块开关影响
                "score_filter": module_enabled("scoring.enabled"),
            }
        )

    def next(self):
        # 记净值：所有账户共用当日行情
        prices = {name: d.close[0] for name, d in self.data_map.items()}
        self.current_date = self.datas[0].datetime.date(0).isoformat()

        for pf in self.accounts.values():
            pf.mark_nav(self.current_date, prices)

        # 先卖：四账户各自风控（互不影响）
        for name, pf in self.accounts.items():
            self._execute_sells(pf)

        # 后买：四账户各自闸门与仓位
        for name, pf in self.accounts.items():
            self._execute_buys(pf, name)

    def stop(self):
        for name, pf in self.accounts.items():
            self._force_close(pf)
            final = pf.daily_nav[-1]["nav"] if pf.daily_nav else 1.0
            print(f"[{name}] 最终净值 {final:.4f} ({(final-1)*100:+.2f}%) "
                  f"交易 {len(pf.trades)} 笔")
