import pandas as pd
import vectorbt as vbt


def sma(close: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average using vectorbt."""
    res = vbt.MA.run(close, window=window, ewm=False)
    out: pd.Series = res.ma
    return out


def ema(close: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average using vectorbt."""
    res = vbt.MA.run(close, window=window, ewm=True)
    out: pd.Series = res.ma
    return out
