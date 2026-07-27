"""Run SMA Crossover backtest demo on Binance OHLCV data."""

import argparse
import sys
from typing import Any

from quantlab.backtest.engine import extract_metrics, run_backtest
from quantlab.data.ohlcv import get_ohlcv
from quantlab.strategies.sma_crossover import generate_signals


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SMA Crossover backtest demo on historical Binance data."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading pair symbol (default: BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="Candle timeframe (default: 1d)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default="2022-01-01",
        help="Start date (default: 2022-01-01)",
    )
    parser.add_argument(
        "--until",
        type=str,
        default="2024-01-01",
        help="End date (default: 2024-01-01)",
    )
    parser.add_argument(
        "--fast-window",
        type=int,
        default=20,
        help="Fast SMA window (default: 20)",
    )
    parser.add_argument(
        "--slow-window",
        type=int,
        default=50,
        help="Slow SMA window (default: 50)",
    )
    parser.add_argument(
        "--init-cash",
        type=float,
        default=10000.0,
        help="Initial cash (default: 10000.0)",
    )
    parser.add_argument(
        "--fees",
        type=float,
        default=0.001,
        help="Fee rate per trade (default: 0.001)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.001,
        help="Slippage rate per trade (default: 0.001)",
    )
    return parser.parse_args(args)


def _format_metric(val: float | int | None, is_percentage: bool = False) -> str:
    if val is None:
        return "N/A"
    if is_percentage and isinstance(val, (int, float)):
        return f"{val * 100:.2f}%"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def print_report(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    print("=" * 45)
    print("      SMA CROSSOVER BACKTEST REPORT          ")
    print("=" * 45)
    print(f"Symbol:         {args.symbol}")
    print(f"Timeframe:      {args.timeframe}")
    print(f"Date Range:     {args.since} to {args.until}")
    print(f"Fast Window:    {args.fast_window}")
    print(f"Slow Window:    {args.slow_window}")
    print(f"Initial Cash:   ${args.init_cash:,.2f}")
    print(f"Fees:           {args.fees * 100:.2f}%")
    print(f"Slippage:       {args.slippage * 100:.2f}%")
    print("-" * 45)
    print(
        f"Total Return:   {_format_metric(metrics.get('total_return'), is_percentage=True)}"
    )
    print(f"Sharpe Ratio:   {_format_metric(metrics.get('sharpe_ratio'))}")
    print(
        f"Max Drawdown:   {_format_metric(metrics.get('max_drawdown'), is_percentage=True)}"
    )
    print(
        f"Win Rate:       {_format_metric(metrics.get('win_rate'), is_percentage=True)}"
    )
    print(f"Total Trades:   {_format_metric(metrics.get('total_trades'))}")
    print("=" * 45)


def main(args: list[str] | None = None) -> dict[str, Any]:
    parsed_args = parse_args(args)
    try:
        df = get_ohlcv(
            symbol=parsed_args.symbol,
            timeframe=parsed_args.timeframe,
            since=parsed_args.since,
            until=parsed_args.until,
        )
        close = df["close"]
        entries, exits = generate_signals(
            close=close,
            fast_window=parsed_args.fast_window,
            slow_window=parsed_args.slow_window,
        )
        portfolio = run_backtest(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=parsed_args.init_cash,
            fees=parsed_args.fees,
            slippage=parsed_args.slippage,
        )
        metrics = extract_metrics(portfolio)
        print_report(parsed_args, metrics)
        return metrics
    except Exception as e:
        print(f"Error executing backtest demo: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
