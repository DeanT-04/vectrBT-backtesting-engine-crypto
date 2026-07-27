"""Unit tests for parameter screening module."""

import sys
import pandas as pd
import pytest
import quantlab.screening.param_sweep as param_sweep_module
from quantlab.screening.param_sweep import run_parameter_sweep


@pytest.fixture
def synthetic_close_series() -> pd.Series:
    """Generate 100 days of synthetic price data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1D", tz="UTC")
    prices = [100.0 + i * 0.5 if i % 2 == 0 else 100.0 - i * 0.3 for i in range(100)]
    return pd.Series(prices, index=dates, name="close")


def test_run_parameter_sweep_synthetic_grid(synthetic_close_series: pd.Series):
    """Verify sweep executes all valid fast < slow combinations and returns expected metrics."""
    fast_windows = [5, 10]
    slow_windows = [20, 30]

    results = run_parameter_sweep(
        close=synthetic_close_series,
        fast_windows=fast_windows,
        slow_windows=slow_windows,
    )

    # 2 fast * 2 slow = 4 valid pairs (5<20, 5<30, 10<20, 10<30)
    assert len(results) == 4

    expected_keys = {
        "fast_window",
        "slow_window",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "total_trades",
    }

    for item in results:
        assert set(item.keys()) == expected_keys
        assert item["fast_window"] < item["slow_window"]
        assert isinstance(item["total_trades"], int)


def test_run_parameter_sweep_skips_invalid_pairs(synthetic_close_series: pd.Series):
    """Verify invalid combinations (fast_window >= slow_window) are silently skipped."""
    fast_windows = [10, 20]
    slow_windows = [10, 15]

    results = run_parameter_sweep(
        close=synthetic_close_series,
        fast_windows=fast_windows,
        slow_windows=slow_windows,
    )

    # Combinations: (10,10)->skip, (10,15)->valid, (20,10)->skip, (20,15)->skip
    assert len(results) == 1
    assert results[0]["fast_window"] == 10
    assert results[0]["slow_window"] == 15


def test_run_parameter_sweep_no_data_import():
    """Verify that param_sweep does not import data module or get_ohlcv."""
    assert not hasattr(param_sweep_module, "get_ohlcv")
    # Verify quantlab.data is not imported by param_sweep module globals
    for attr in dir(param_sweep_module):
        val = getattr(param_sweep_module, attr)
        if hasattr(val, "__name__"):
            assert "quantlab.data" not in getattr(val, "__name__", "")
