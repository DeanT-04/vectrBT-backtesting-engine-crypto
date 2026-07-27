"""Unit tests for SMA crossover strategy signal generation."""

import pandas as pd
import pytest
from quantlab.strategies.sma_crossover import generate_signals


def test_sma_crossover_known_signals() -> None:
    """Test crossover up and crossover down at exact known index positions."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    # Prices designed so fast (SMA-2) crosses above slow (SMA-4) at index 4
    # and fast crosses below slow at index 7.
    prices = [10.0, 10.0, 10.0, 10.0, 15.0, 20.0, 15.0, 10.0, 5.0, 5.0]
    close = pd.Series(prices, index=dates)

    entries, exits = generate_signals(close, fast_window=2, slow_window=4)

    expected_entries = pd.Series(
        [False, False, False, False, True, False, False, False, False, False],
        index=dates,
    )
    expected_exits = pd.Series(
        [False, False, False, False, False, False, False, True, False, False],
        index=dates,
    )

    pd.testing.assert_series_equal(entries, expected_entries)
    pd.testing.assert_series_equal(exits, expected_exits)


def test_invalid_window_raises_exception() -> None:
    """Test that fast_window >= slow_window raises a ValueError."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=dates)

    with pytest.raises(ValueError, match="must be strictly less than"):
        generate_signals(close, fast_window=5, slow_window=5)

    with pytest.raises(ValueError, match="must be strictly less than"):
        generate_signals(close, fast_window=10, slow_window=5)


def test_nan_warmup_period() -> None:
    """Test that warm-up rows (where SMAs are NaN) have False entries and exits, not NaN."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    close = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0], index=dates)

    fast_window = 2
    slow_window = 4
    entries, exits = generate_signals(close, fast_window=fast_window, slow_window=slow_window)

    # Warm-up period for slow SMA window = 4 is first 4 rows (index 0 to 3)
    warmup_entries = entries.iloc[:slow_window]
    warmup_exits = exits.iloc[:slow_window]

    assert warmup_entries.dtype == bool
    assert warmup_exits.dtype == bool
    assert not warmup_entries.isna().any()
    assert not warmup_exits.isna().any()
    assert not warmup_entries.any()
    assert not warmup_exits.any()


def test_flat_price_series_no_signals() -> None:
    """Test a flat price series produces all False entries and exits without error."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    close = pd.Series([100.0] * 10, index=dates)

    entries, exits = generate_signals(close, fast_window=2, slow_window=4)

    assert entries.dtype == bool
    assert exits.dtype == bool
    assert not entries.any()
    assert not exits.any()
