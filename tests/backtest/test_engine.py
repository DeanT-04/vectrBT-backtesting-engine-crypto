import numpy as np
import pandas as pd
import pytest
from quantlab.backtest.engine import extract_metrics, run_backtest, run_buy_and_hold_benchmark


def test_simple_profitable_trade() -> None:
    """Test a simple profitable trade scenario with 1 trade and positive return."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=dates)
    entries = pd.Series([True, False, False, False, False], index=dates)
    exits = pd.Series([False, False, False, True, False], index=dates)

    portfolio = run_backtest(close, entries, exits, init_cash=10000.0, fees=0.001, slippage=0.001)
    metrics = extract_metrics(portfolio)

    assert metrics["total_trades"] == 1
    assert metrics["total_return"] is not None
    assert metrics["total_return"] > 0.0


def test_mismatched_index_raises_exception() -> None:
    """Test that mismatched indices between close, entries, or exits raise ValueError."""
    dates1 = pd.date_range("2023-01-01", periods=5, freq="D")
    dates2 = pd.date_range("2023-01-02", periods=5, freq="D")

    close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=dates1)
    entries = pd.Series([True, False, False, False, False], index=dates1)
    exits_mismatched = pd.Series([False, False, False, True, False], index=dates2)

    with pytest.raises(ValueError, match="same index"):
        run_backtest(close, entries, exits_mismatched)

    entries_mismatched = pd.Series([True, False, False, False, False], index=dates2)
    exits = pd.Series([False, False, False, True, False], index=dates1)

    with pytest.raises(ValueError, match="same index"):
        run_backtest(close, entries_mismatched, exits)


def test_zero_fee_vs_fee_impact() -> None:
    """Test that zero-fee run achieves a higher total_return than a non-zero-fee run."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=dates)
    entries = pd.Series([True, False, False, False, False], index=dates)
    exits = pd.Series([False, False, False, True, False], index=dates)

    pf_no_fee = run_backtest(close, entries, exits, init_cash=10000.0, fees=0.0, slippage=0.0)
    pf_with_fee = run_backtest(close, entries, exits, init_cash=10000.0, fees=0.005, slippage=0.005)

    metrics_no_fee = extract_metrics(pf_no_fee)
    metrics_with_fee = extract_metrics(pf_with_fee)

    assert metrics_no_fee["total_return"] is not None
    assert metrics_with_fee["total_return"] is not None
    assert metrics_no_fee["total_return"] > metrics_with_fee["total_return"]


def test_no_entries_scenario() -> None:
    """Test a scenario with no entry signals at all."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=dates)
    entries = pd.Series([False] * 5, index=dates)
    exits = pd.Series([False] * 5, index=dates)

    portfolio = run_backtest(close, entries, exits)
    metrics = extract_metrics(portfolio)

    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] is None


def test_metrics_return_plain_python_types() -> None:
    """Test that all returned metrics are plain Python floats, ints, or None."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=dates)
    entries = pd.Series([True, False, False, False, False], index=dates)
    exits = pd.Series([False, False, False, True, False], index=dates)

    portfolio = run_backtest(close, entries, exits)
    metrics = extract_metrics(portfolio)

    for key, value in metrics.items():
        assert type(value) in (float, int, type(None)), f"Key {key} has non-plain type {type(value)}"

    # Also test for empty entries scenario to verify types when values are None
    entries_empty = pd.Series([False] * 5, index=dates)
    exits_empty = pd.Series([False] * 5, index=dates)
    portfolio_empty = run_backtest(close, entries_empty, exits_empty)
    metrics_empty = extract_metrics(portfolio_empty)

    for key, value in metrics_empty.items():
        assert type(value) in (float, int, type(None)), f"Key {key} has non-plain type {type(value)}"


def test_buy_and_hold_benchmark_known_prices() -> None:
    """Test buy-and-hold return on a synthetic price series with known start and end price."""
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    close = pd.Series([100.0, 200.0], index=dates)
    fees = 0.001
    slippage = 0.001

    metrics = run_buy_and_hold_benchmark(close, init_cash=10000.0, fees=fees, slippage=slippage)

    expected_return = (1.0 - fees) ** 2 * ((1.0 - slippage) / (1.0 + slippage)) * (200.0 / 100.0) - 1.0

    assert metrics["total_trades"] == 1
    assert metrics["total_return"] is not None
    assert pytest.approx(metrics["total_return"], rel=1e-4) == expected_return


def test_buy_and_hold_period_end_valuation() -> None:
    """Confirm position is correctly valued at period end (constraint-3 resolution verification)."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 110.0, 120.0, 130.0, 150.0], index=dates)

    metrics = run_buy_and_hold_benchmark(close, init_cash=10000.0, fees=0.001, slippage=0.001)

    assert metrics["total_trades"] == 1
    assert metrics["total_return"] is not None
    assert metrics["total_return"] > 0.45
    assert metrics["win_rate"] == 1.0
    assert metrics["sharpe_ratio"] is not None

