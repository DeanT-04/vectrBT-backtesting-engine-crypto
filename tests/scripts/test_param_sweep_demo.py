"""Tests for param_sweep_demo script."""

from unittest.mock import patch
import pandas as pd
import pytest
from scripts.param_sweep_demo import main


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
    """Verify main runs successfully with mocked get_ohlcv and returns sweep results list."""
    with patch("scripts.param_sweep_demo.get_ohlcv", return_value=synthetic_ohlcv_df) as mock_get:
        results = main(["--fast-windows", "5", "10", "--slow-windows", "20", "30"])
        mock_get.assert_called_once_with(
            symbol="BTC/USDT", timeframe="1d", since="2022-01-01", until="2024-01-01"
        )
        assert isinstance(results, list)
        assert len(results) == 4

        captured = capsys.readouterr()
        assert "SMA CROSSOVER PARAMETER SCREENING REPORT" in captured.out
        assert "NOTE: This is an exploratory, in-sample sweep" in captured.out
        assert "Best by Sharpe:" in captured.out
        assert "Worst by Sharpe:" in captured.out


def test_main_calls_run_parameter_sweep(synthetic_ohlcv_df: pd.DataFrame):
    """Verify main calls run_parameter_sweep from the screening library module."""
    mock_results = [
        {
            "fast_window": 5,
            "slow_window": 20,
            "total_return": 0.1,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
            "win_rate": 0.6,
            "total_trades": 4,
        }
    ]
    with patch("scripts.param_sweep_demo.get_ohlcv", return_value=synthetic_ohlcv_df):
        with patch("scripts.param_sweep_demo.run_parameter_sweep", return_value=mock_results) as mock_sweep:
            results = main(["--fast-windows", "5", "--slow-windows", "20"])
            mock_sweep.assert_called_once()
            assert results == mock_results


def test_main_get_ohlcv_exception_handled(capsys: pytest.CaptureFixture[str]):
    """Verify main handles get_ohlcv exceptions gracefully with non-zero exit code."""
    with patch(
        "scripts.param_sweep_demo.get_ohlcv",
        side_effect=ValueError("Binance returned no OHLCV data for BTC/USDT."),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "Error executing parameter sweep demo: Binance returned no OHLCV data" in captured.err
