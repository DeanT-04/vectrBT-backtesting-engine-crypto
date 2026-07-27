"""Parameter screening module for sensitivity sweeps across strategy parameter grids."""

import pandas as pd
from quantlab.backtest.engine import extract_metrics, run_backtest
from quantlab.strategies.sma_crossover import generate_signals


def run_parameter_sweep(
    close: pd.Series,
    fast_windows: list[int],
    slow_windows: list[int],
    init_cash: float = 10_000.0,
    fees: float = 0.001,
    slippage: float = 0.001,
) -> list[dict]:
    """Run parameter sensitivity sweep across fast and slow SMA window pairs.

    Args:
        close: Pandas Series of close prices.
        fast_windows: List of fast window lookbacks to evaluate.
        slow_windows: List of slow window lookbacks to evaluate.
        init_cash: Initial cash for portfolio simulation. Default 10000.0.
        fees: Fee rate per trade. Default 0.001 (0.1%).
        slippage: Slippage rate per trade. Default 0.001 (0.1%).

    Returns:
        list[dict]: List of dictionaries containing fast_window, slow_window,
            and all performance metrics extracted from extract_metrics.
    """
    results: list[dict] = []
    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:
                continue

            entries, exits = generate_signals(
                close=close, fast_window=fast, slow_window=slow
            )
            portfolio = run_backtest(
                close=close,
                entries=entries,
                exits=exits,
                init_cash=init_cash,
                fees=fees,
                slippage=slippage,
            )
            metrics = extract_metrics(portfolio)

            result = {
                "fast_window": fast,
                "slow_window": slow,
                **metrics,
            }
            results.append(result)

    return results
