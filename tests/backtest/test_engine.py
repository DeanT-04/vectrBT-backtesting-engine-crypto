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


def test_closed_before_end_is_no_op() -> None:
    """Test that a strategy closing before the final bar is unaffected by force-close (no phantom trades or double fees)."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=dates)
    entries = pd.Series([True, False, False, False, False], index=dates)
    exits = pd.Series([False, False, True, False, False], index=dates)  # Exit at bar 2

    portfolio = run_backtest(close, entries, exits, init_cash=10000.0, fees=0.001, slippage=0.001)
    metrics = extract_metrics(portfolio)

    assert metrics["total_trades"] == 1
    assert metrics["win_rate"] == 1.0
    # Expected return for trade entered at 100 on bar 0 and exited at 110 on bar 2
    expected_return = (1.0 - 0.001) ** 2 * ((1.0 - 0.001) / (1.0 + 0.001)) * (110.0 / 100.0) - 1.0
    assert metrics["total_return"] is not None
    assert pytest.approx(metrics["total_return"], rel=1e-4) == expected_return


def test_position_still_open_at_end_gets_force_closed() -> None:
    """Test that an open position at the final bar is force-closed, charging exit fees and counting as a completed trade."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=dates)
    entries = pd.Series([True, False, False, False, False], index=dates)
    exits = pd.Series([False, False, False, False, False], index=dates)  # Never exited by strategy

    pf_with_fees = run_backtest(close, entries, exits, init_cash=10000.0, fees=0.001, slippage=0.001)
    pf_zero_fees = run_backtest(close, entries, exits, init_cash=10000.0, fees=0.0, slippage=0.0)

    metrics_with_fees = extract_metrics(pf_with_fees)
    metrics_zero_fees = extract_metrics(pf_zero_fees)

    # Position must be counted as a completed trade and win_rate must be calculated
    assert metrics_with_fees["total_trades"] == 1
    assert metrics_with_fees["win_rate"] == 1.0

    # Exit fees must be charged, so zero fees return must be strictly greater than fee-bearing return
    assert metrics_zero_fees["total_return"] is not None
    assert metrics_with_fees["total_return"] is not None
    assert metrics_zero_fees["total_return"] > metrics_with_fees["total_return"]


def test_single_bar_series_no_forced_exit() -> None:
    """Test single-bar edge case (len(close) == 1): no forced exit added on a single-bar series."""
    dates = pd.date_range("2023-01-01", periods=1, freq="D")
    close = pd.Series([100.0], index=dates)
    entries = pd.Series([True], index=dates)
    exits = pd.Series([False], index=dates)

    portfolio = run_backtest(close, entries, exits, init_cash=10000.0, fees=0.001, slippage=0.001)
    metrics = extract_metrics(portfolio)

    # Position is opened on bar 0 but not force-closed on bar 0 (preventing entry and exit cancelling on same bar)
    assert metrics["total_trades"] == 1
    assert metrics["win_rate"] == 0.0


def test_real_cached_data_no_op_and_open_position_drift() -> None:
    """Test using real cached data for both already-closed (no-op) and open position scenarios.

    Verifies that:
    1. When a position is already closed before the final bar (e.g. Fast=5, Slow=20),
       run_backtest produces a bit-for-bit identical total_return compared to raw
       vbt.Portfolio.from_signals without forced close.
    2. When a position remains open at the final bar (e.g. Fast=20, Slow=50), trade
       counts remain identical (because vectorbt counts open trade records), but
       run_backtest force-closes the position on the final bar, incurring exit fees/slippage
       and shifting total_return as expected.
    """
    import vectorbt as vbt
    from quantlab.data.ohlcv import get_ohlcv
    from quantlab.strategies.sma_crossover import generate_signals

    df = get_ohlcv("BTC/USDT", "1d", "2022-01-01", "2024-01-01")
    close = df["close"]

    # 1. Closed position before final bar: Fast=5, Slow=20
    entries_5_20, exits_5_20 = generate_signals(close, 5, 20)
    pf_old_5_20 = vbt.Portfolio.from_signals(
        close=close, entries=entries_5_20, exits=exits_5_20, init_cash=10000.0, fees=0.001, slippage=0.001
    )
    pf_new_5_20 = run_backtest(
        close=close, entries=entries_5_20, exits=exits_5_20, init_cash=10000.0, fees=0.001, slippage=0.001
    )
    # Already-closed position: total_return must be bit-for-bit identical (exact float equality)
    assert pf_new_5_20.total_return() == pf_old_5_20.total_return()

    # 2. Open position at final bar: Fast=20, Slow=50
    entries_20_50, exits_20_50 = generate_signals(close, 20, 50)
    pf_old_20_50 = vbt.Portfolio.from_signals(
        close=close, entries=entries_20_50, exits=exits_20_50, init_cash=10000.0, fees=0.001, slippage=0.001
    )
    pf_new_20_50 = run_backtest(
        close=close, entries=entries_20_50, exits=exits_20_50, init_cash=10000.0, fees=0.001, slippage=0.001
    )
    # Trade counts are identical because vectorbt counts both open and closed trade records
    assert pf_new_20_50.trades.count() == pf_old_20_50.trades.count() == 9
    # Position was open at final bar, so forced-close charges exit fees/slippage on final bar, causing total_return to shift
    assert pf_new_20_50.total_return() != pf_old_20_50.total_return()




