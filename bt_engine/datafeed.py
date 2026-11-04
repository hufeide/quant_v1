"""backtrader 数据适配器：把 parquet DataFrame 转成 PandasData

需要为 DataFrame 添加 datetime 索引，并确保字段名符合 backtrader 规范。
"""
import backtrader as bt
import pandas as pd


class StockDataFeed(bt.feeds.PandasData):
    """扩展 PandasData，支持额外字段"""

    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        ('amount', 'amount'),
        ('pct_chg', 'pct_chg'),
        ('turnover_rate', 'turnover_rate'),
        ('market_cap_float', 'market_cap_float'),
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """准备 backtrader 需要的 DataFrame

    Args:
        df: 原始日线 DataFrame，需含 date 列

    Returns:
        datetime 索引的 DataFrame
    """
    df = df.copy()
    if 'date' not in df.columns:
        raise ValueError("DataFrame 必须包含 date 列")

    # 设置 datetime 索引
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime')

    # 确保必需字段存在
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    return df


def add_data_feed(cerebro: bt.Cerebro, code: str, daily_dir: str,
                  fromdate=None, todate=None):
    """为 Cerebro 添加单只股票的 data feed"""
    from src.data_loader import load_daily_with_indicators

    df = load_daily_with_indicators(code, daily_dir)

    if fromdate is not None:
        df = df[df['date'] >= fromdate]
    if todate is not None:
        df = df[df['date'] <= todate]

    if df.empty:
        return None

    df_bt = prepare_dataframe(df)

    data = StockDataFeed(dataname=df_bt, name=code)
    cerebro.adddata(data, name=code)
    return data
