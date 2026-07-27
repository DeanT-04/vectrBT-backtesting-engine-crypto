"""SMA Crossover strategy signal generation."""

import pandas as pd
from quantlab.indicators.trend import sma


def generate_signals(
    close: pd.Series, fast_window: int, slow_window: int
) -> tuple[pd.Series, pd.Series]:
    """Generate entry and exit signals based on Simple Moving Average crossover.

    Args:
        close: Pandas Series of closing prices.
        fast_window: Lookback period for fast moving average.
        slow_window: Lookback period for slow moving average.

    Returns:
        tuple[pd.Series, pd.Series]: (entries, exits) as boolean Series aligned with close.

    Raises:
        ValueError: If fast_window >= slow_window.
    """
    if fast_window >= slow_window:
        raise ValueError(
            f"fast_window ({fast_window}) must be strictly less than slow_window ({slow_window})."
        )

    fast = sma(close, fast_window)
    slow = sma(close, slow_window)

    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)

    valid = ~(fast.isna() | slow.isna() | prev_fast.isna() | prev_slow.isna())

    raw_entries = (prev_fast <= prev_slow) & (fast > slow)
    raw_exits = (prev_fast >= prev_slow) & (fast < slow)

    entries = (valid & raw_entries).fillna(False).astype(bool)
    exits = (valid & raw_exits).fillna(False).astype(bool)

    return entries, exits
