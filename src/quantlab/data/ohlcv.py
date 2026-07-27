import re
from pathlib import Path
import ccxt
import pandas as pd


def get_ohlcv(
    symbol: str,
    timeframe: str,
    since: str,
    until: str,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance and cache it locally as parquet.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. "BTC/USDT".
    timeframe : str
        Candle timeframe, e.g. "1d", "1h".
    since : str
        ISO date string for start of range, e.g. "2023-01-01".
    until : str
        ISO date string for end of range, e.g. "2023-01-05".
    cache_dir : str
        Directory to store cached parquet files. Defaults to "data".

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by UTC timestamp with columns: open, high, low, close, volume.
    """
    sanitized_symbol = re.sub(r'[/\\:\*\?"<>\|]', "_", symbol)
    cache_filename = f"binance_{sanitized_symbol}_{timeframe}_{since}_{until}.parquet"
    cache_path = Path(cache_dir) / cache_filename

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    since_dt = pd.to_datetime(since, utc=True)
    until_dt = pd.to_datetime(until, utc=True)
    since_ms = int(since_dt.timestamp() * 1000)
    until_ms = int(until_dt.timestamp() * 1000)

    if until_ms < since_ms:
        raise ValueError(f"until date ('{until}') cannot be earlier than since date ('{since}').")

    exchange = ccxt.binance({"enableRateLimit": True})

    all_ohlcv = []
    current_since = since_ms

    while current_since <= until_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=1000)
        if not candles:
            break

        if not all_ohlcv:
            first_candle_ts = candles[0][0]
            if first_candle_ts > since_ms:
                earliest_dt = pd.to_datetime(first_candle_ts, unit="ms", utc=True)
                earliest_str = earliest_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                raise ValueError(
                    f"Binance cannot cover requested range for {symbol}. "
                    f"Requested since '{since}', but earliest available candle is '{earliest_str}'."
                )

        last_ts = candles[-1][0]
        all_ohlcv.extend(candles)

        if last_ts >= until_ms:
            break

        next_since = last_ts + 1
        if next_since <= current_since:
            break
        current_since = next_since

    if not all_ohlcv:
        raise ValueError(f"Binance returned no OHLCV data for {symbol}.")

    last_candle_ts = all_ohlcv[-1][0]
    if last_candle_ts < until_ms:
        latest_dt = pd.to_datetime(last_candle_ts, unit="ms", utc=True)
        latest_str = latest_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        raise ValueError(
            f"Binance cannot cover requested range for {symbol}. "
            f"Requested until '{until}', but latest available candle is '{latest_str}'."
        )

    filtered_ohlcv = [c for c in all_ohlcv if since_ms <= c[0] <= until_ms]

    df = pd.DataFrame(filtered_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

    return df
