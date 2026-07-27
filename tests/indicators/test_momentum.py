import pandas as pd
import pytest
from quantlab.indicators.momentum import rsi


@pytest.fixture
def wave_prices() -> pd.Series:
    """Fixture providing price series with clear up-then-down pattern."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    prices = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 18.0, 16.0, 14.0, 12.0,
              10.0, 8.0, 6.0, 4.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    return pd.Series(prices, index=dates)


def test_rsi_bounds(wave_prices: pd.Series) -> None:
    result = rsi(wave_prices, window=5)
    valid_values = result.dropna()

    assert not valid_values.empty
    assert (valid_values >= 0.0).all()
    assert (valid_values <= 100.0).all()


def test_rsi_returns_series_with_same_index(wave_prices: pd.Series) -> None:
    result = rsi(wave_prices, window=5)
    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, wave_prices.index)


def test_rsi_default_window(wave_prices: pd.Series) -> None:
    result_default = rsi(wave_prices)
    result_explicit_14 = rsi(wave_prices, window=14)

    pd.testing.assert_series_equal(result_default, result_explicit_14)


def test_rsi_window_larger_than_data(wave_prices: pd.Series) -> None:
    # Window larger than data length (25 > 20)
    result = rsi(wave_prices, window=25)
    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, wave_prices.index)
    # When window > len(data), vectorbt returns all NaNs
    assert result.isna().all()
