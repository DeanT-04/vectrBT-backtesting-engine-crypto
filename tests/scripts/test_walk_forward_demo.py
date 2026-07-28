"""Tests for walk_forward_demo script."""

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from scripts.walk_forward_demo import main, resolve_window_list


@pytest.fixture
def synthetic_ohlcv_2yr_df() -> pd.DataFrame:
    """Generate 2 years of synthetic OHLCV data."""
    dates = pd.date_range("2018-01-01", "2020-01-01", freq="1D", tz="UTC")
    np.random.seed(42)
    x = np.linspace(0, 10 * np.pi, len(dates))
    prices = 100.0 + 10.0 * np.sin(x) + np.linspace(0, 50, len(dates))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": [1000.0] * len(dates),
        },
        index=dates,
    )


def test_resolve_window_list():
    """Test resolution of window lists from explicit list vs range arguments."""
    # List takes precedence if provided and non-empty
    assert resolve_window_list([5, 10], None, None, None, [1, 2]) == [5, 10]

    # Range parameters
    assert resolve_window_list(None, 5, 20, 5, [1, 2]) == [5, 10, 15, 20]
    assert resolve_window_list(None, 10, 30, 10, [1, 2]) == [10, 20, 30]

    # Fallback to default
    assert resolve_window_list(None, None, None, None, [1, 2]) == [1, 2]


def test_main_success(
    synthetic_ohlcv_2yr_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]
):
    """Verify main runs successfully with mocked get_ohlcv and returns aggregation dict."""
    with patch(
        "scripts.walk_forward_demo.get_ohlcv", return_value=synthetic_ohlcv_2yr_df
    ) as mock_get:
        agg = main(
            [
                "--pool-start",
                "2018-01-01",
                "--pool-end",
                "2020-01-01",
                "--fast-start",
                "5",
                "--fast-stop",
                "10",
                "--fast-step",
                "5",
                "--slow-start",
                "20",
                "--slow-stop",
                "30",
                "--slow-step",
                "10",
                "--min-trades",
                "1",
            ]
        )

        mock_get.assert_called_once_with(
            symbol="BTC/USDT",
            timeframe="1d",
            since="2018-01-01",
            until="2020-01-01",
        )

        assert isinstance(agg, dict)
        assert agg["folds_total"] == 2
        assert agg["folds_evaluated"] >= 1

        captured = capsys.readouterr()
        assert "WALK-FORWARD VALIDATION REPORT" in captured.out
        assert "WALK-FORWARD AGGREGATE SUMMARY" in captured.out
        assert "Generated 2 fold(s)." in captured.out


def test_main_get_ohlcv_exception_handled(capsys: pytest.CaptureFixture[str]):
    """Verify main handles get_ohlcv exceptions gracefully with non-zero exit code."""
    with patch(
        "scripts.walk_forward_demo.get_ohlcv",
        side_effect=ValueError("Binance error."),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--pool-start",
                    "2018-01-01",
                    "--pool-end",
                    "2020-01-01",
                ]
            )
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "Error executing walk-forward validation demo: Binance error." in captured.err
