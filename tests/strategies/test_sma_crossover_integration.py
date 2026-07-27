"""Integration test connecting signals, indicators, and backtest engine."""

import pandas as pd
from quantlab.backtest.engine import extract_metrics, run_backtest
from quantlab.strategies.sma_crossover import generate_signals


def test_sma_crossover_end_to_end_integration() -> None:
    """Test full pipeline: synthetic prices -> sma_crossover signals -> backtest engine -> metrics."""
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    prices = [
        10.0, 10.0, 10.0, 10.0, 15.0, 20.0, 25.0, 30.0,
        25.0, 20.0, 15.0, 10.0, 5.0, 5.0, 5.0,
    ]
    close = pd.Series(prices, index=dates)

    entries, exits = generate_signals(close, fast_window=2, slow_window=4)
    portfolio = run_backtest(close, entries, exits)
    metrics = extract_metrics(portfolio)

    assert metrics["total_trades"] >= 1
    for key, value in metrics.items():
        assert type(value) in (float, int, type(None)), f"Key {key} has non-plain type {type(value)}"
