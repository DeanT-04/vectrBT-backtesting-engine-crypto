"""Backtest engine wrapping vectorbt Portfolio simulation."""

from typing import Any
import pandas as pd
import vectorbt as vbt


def run_backtest(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
    slippage: float = 0.001,
) -> vbt.Portfolio:
    """Run a vectorbt backtest from price and entry/exit signals.

    Args:
        close: Pandas Series of close prices.
        entries: Boolean Series of entry signals aligned with close index.
        exits: Boolean Series of exit signals aligned with close index.
        init_cash: Initial cash for portfolio simulation. Default 10000.0.
        fees: Fee rate per trade. Default 0.001 (0.1%).
        slippage: Slippage rate per trade. Default 0.001 (0.1%).

    Returns:
        vbt.Portfolio object containing simulation results.

    Raises:
        ValueError: If close, entries, and exits do not share the exact same index.
    """
    if not close.index.equals(entries.index) or not close.index.equals(exits.index):
        raise ValueError("close, entries, and exits must share the same index.")

    return vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
    )


def extract_metrics(portfolio: vbt.Portfolio) -> dict[str, Any]:
    """Extract standard performance metrics from a vectorbt Portfolio object.

    Args:
        portfolio: A vectorbt Portfolio object.

    Returns:
        dict containing standard backtest metrics as plain Python types:
            - total_return (float or None)
            - sharpe_ratio (float or None)
            - max_drawdown (float or None)
            - win_rate (float or None)
            - total_trades (int)

        If a metric is undefined for a given result (e.g. no trades taken, so
        win_rate is undefined or NaN), None is returned for that metric key
        rather than crashing or returning NaN.
    """
    total_trades_raw = portfolio.trades.count()
    total_trades = int(total_trades_raw) if not pd.isna(total_trades_raw) else 0

    def _to_float(val: Any) -> float | None:
        if val is None or pd.isna(val):
            return None
        return float(val)

    total_return = _to_float(portfolio.total_return())
    sharpe_ratio = _to_float(portfolio.sharpe_ratio())
    max_drawdown = _to_float(portfolio.max_drawdown())
    win_rate = _to_float(portfolio.trades.win_rate())

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": total_trades,
    }
