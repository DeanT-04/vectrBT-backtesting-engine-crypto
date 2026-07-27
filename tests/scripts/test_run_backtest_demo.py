"""Tests for run_backtest_demo script."""

from unittest.mock import patch
import pandas as pd
import pytest
from scripts.run_backtest_demo import main


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    """Generate 100 days of synthetic OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1D", tz="UTC")
    prices = [100.0 + i * 0.5 if i % 2 == 0 else 100.0 - i * 0.3 for i in range(100)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": [1000.0] * 100,
        },
        index=dates,
    )


def test_main_success(synthetic_ohlcv_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]):
    """Verify main runs successfully with mocked get_ohlcv and returns metrics dict."""
    with patch("scripts.run_backtest_demo.get_ohlcv", return_value=synthetic_ohlcv_df) as mock_get:
        metrics = main(["--fast-window", "5", "--slow-window", "10"])
        mock_get.assert_called_once_with(
            symbol="BTC/USDT", timeframe="1d", since="2022-01-01", until="2024-01-01"
        )
        assert isinstance(metrics, dict)
        expected_keys = {"total_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"}
        assert expected_keys.issubset(metrics.keys())

        captured = capsys.readouterr()
        assert "SMA CROSSOVER BACKTEST REPORT" in captured.out
        assert "Symbol:         BTC/USDT" in captured.out


def test_main_get_ohlcv_exception_handled(capsys: pytest.CaptureFixture[str]):
    """Verify main handles get_ohlcv exceptions gracefully with non-zero exit code."""
    with patch(
        "scripts.run_backtest_demo.get_ohlcv",
        side_effect=ValueError("Binance returned no OHLCV data for BTC/USDT."),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "Error executing backtest demo: Binance returned no OHLCV data" in captured.err


def test_main_invalid_windows_exception_handled(
    synthetic_ohlcv_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]
):
    """Verify main handles strategy validation errors (fast_window >= slow_window) gracefully."""
    with patch("scripts.run_backtest_demo.get_ohlcv", return_value=synthetic_ohlcv_df):
        with pytest.raises(SystemExit) as exc_info:
            main(["--fast-window", "50", "--slow-window", "20"])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "Error executing backtest demo:" in captured.err
        assert "fast_window (50) must be strictly less than slow_window (20)" in captured.err
