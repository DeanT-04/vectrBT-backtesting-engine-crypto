from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from quantlab.data.ohlcv import get_ohlcv


@pytest.fixture
def mock_binance():
    with patch("quantlab.data.ohlcv.ccxt.binance") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


def test_get_ohlcv_full_coverage(mock_binance, tmp_path):
    # Setup mock data for 2023-01-01 to 2023-01-03
    mock_candles = [
        [1672531200000, 100.0, 105.0, 95.0, 102.0, 1000.0],  # 2023-01-01 00:00:00 UTC
        [1672617600000, 102.0, 108.0, 101.0, 107.0, 1200.0], # 2023-01-02 00:00:00 UTC
        [1672704000000, 107.0, 110.0, 106.0, 109.0, 1100.0], # 2023-01-03 00:00:00 UTC
    ]
    mock_binance.fetch_ohlcv.return_value = mock_candles

    cache_dir = str(tmp_path)
    df = get_ohlcv("BTC/USDT", "1d", "2023-01-01", "2023-01-03", cache_dir=cache_dir)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert str(df.index.tz) == "UTC"
    assert df.iloc[0]["open"] == 100.0
    assert df.iloc[-1]["close"] == 109.0

    expected_cache_file = tmp_path / "binance_BTC_USDT_1d_2023-01-01_2023-01-03.parquet"
    assert expected_cache_file.exists()
    assert mock_binance.fetch_ohlcv.called


def test_get_ohlcv_cached_data_second_call(mock_binance, tmp_path):
    mock_candles = [
        [1672531200000, 100.0, 105.0, 95.0, 102.0, 1000.0],
        [1672617600000, 102.0, 108.0, 101.0, 107.0, 1200.0],
    ]
    mock_binance.fetch_ohlcv.return_value = mock_candles

    cache_dir = str(tmp_path)
    # First call: fetches and caches
    df1 = get_ohlcv("BTC/USDT", "1d", "2023-01-01", "2023-01-02", cache_dir=cache_dir)
    assert mock_binance.fetch_ohlcv.call_count == 1

    # Reset mock to verify second call doesn't touch exchange
    mock_binance.fetch_ohlcv.reset_mock()

    # Second call: returns cached data
    df2 = get_ohlcv("BTC/USDT", "1d", "2023-01-01", "2023-01-02", cache_dir=cache_dir)
    mock_binance.fetch_ohlcv.assert_not_called()
    pd.testing.assert_frame_equal(df1, df2)


def test_get_ohlcv_earliest_candle_after_since(mock_binance, tmp_path):
    # Requested since="2023-01-01", but earliest available candle is 2023-01-02
    mock_candles = [
        [1672617600000, 102.0, 108.0, 101.0, 107.0, 1200.0], # 2023-01-02 00:00:00 UTC
        [1672704000000, 107.0, 110.0, 106.0, 109.0, 1100.0],
    ]
    mock_binance.fetch_ohlcv.return_value = mock_candles

    cache_dir = str(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        get_ohlcv("BTC/USDT", "1d", "2023-01-01", "2023-01-03", cache_dir=cache_dir)

    assert "Binance cannot cover requested range for BTC/USDT" in str(exc_info.value)
    assert "2023-01-02" in str(exc_info.value)
