import pandas as pd
import vectorbt as vbt


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index using vectorbt."""
    res = vbt.RSI.run(close, window=window)
    out: pd.Series = res.rsi
    return out
