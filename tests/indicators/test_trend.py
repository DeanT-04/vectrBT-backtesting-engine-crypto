import pandas as pd
import pytest
from quantlab.indicators.trend import ema, sma


@pytest.fixture
def linear_prices() -> pd.Series:
    """Fixture providing 20 sequential price points (1.0 to 20.0)."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    return pd.Series([float(i) for i in range(1, 21)], index=dates)


def test_sma_values(linear_prices: pd.Series) -> None:
    window = 5
    result = sma(linear_prices, window=window)

    # Calculate expected SMA via pandas rolling
    expected = linear_prices.rolling(window).mean()

    # First window - 1 elements should be NaN
    assert result.iloc[: window - 1].isna().all()
    # Check non-NaN values match expected
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_ema_values(linear_prices: pd.Series) -> None:
    window = 5
    result = ema(linear_prices, window=window)

    # Calculate expected EMA via pandas ewm with adjust=False
    expected = linear_prices.ewm(span=window, min_periods=window, adjust=False).mean()

    # First window - 1 elements should be NaN
    assert result.iloc[: window - 1].isna().all()
    # Check non-NaN values match expected
    pd.testing.assert_series_equal(result, expected, check_names=False)



def test_sma_returns_series_with_same_index(linear_prices: pd.Series) -> None:
    result = sma(linear_prices, window=5)
    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, linear_prices.index)


def test_ema_returns_series_with_same_index(linear_prices: pd.Series) -> None:
    result = ema(linear_prices, window=5)
    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, linear_prices.index)


def test_sma_window_larger_than_data(linear_prices: pd.Series) -> None:
    # Window larger than data length (25 > 20)
    result = sma(linear_prices, window=25)
    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, linear_prices.index)
    # When window > len(data), vectorbt returns all NaNs
    assert result.isna().all()


def test_ema_window_larger_than_data(linear_prices: pd.Series) -> None:
    # Window larger than data length (25 > 20)
    result = ema(linear_prices, window=25)
    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, linear_prices.index)
    # When window > len(data), vectorbt returns all NaNs
    assert result.isna().all()
