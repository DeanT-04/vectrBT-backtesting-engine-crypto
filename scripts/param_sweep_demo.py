"""Run SMA Crossover parameter sensitivity sweep on Binance OHLCV data."""

import argparse
import sys
from typing import Any

from quantlab.data.ohlcv import get_ohlcv
from quantlab.screening.param_sweep import run_parameter_sweep


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SMA Crossover parameter sweep on historical Binance data."
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
        "--fast-windows",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="Fast SMA window lookbacks (default: 5 10 20 50)",
    )
    parser.add_argument(
        "--slow-windows",
        type=int,
        nargs="+",
        default=[20, 50, 100, 200],
        help="Slow SMA window lookbacks (default: 20 50 100 200)",
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


def print_report(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    print("=" * 75)
    print("           SMA CROSSOVER PARAMETER SCREENING REPORT          ")
    print("=" * 75)
    print(f"Symbol:         {args.symbol}")
    print(f"Timeframe:      {args.timeframe}")
    print(f"Date Range:     {args.since} to {args.until}")
    print(f"Initial Cash:   ${args.init_cash:,.2f}")
    print(f"Fees:           {args.fees * 100:.2f}%")
    print(f"Slippage:       {args.slippage * 100:.2f}%")
    print("-" * 75)
    print(
        "NOTE: This is an exploratory, in-sample sweep across historical data. "
        "Results should not be used for live trading without out-of-sample validation."
    )
    print("-" * 75)

    if not results:
        print("No valid parameter combinations evaluated.")
        print("=" * 75)
        return

    # Sort results by Sharpe ratio descending (placing N/A at the end)
    sorted_results = sorted(
        results,
        key=lambda r: (
            r["sharpe_ratio"] is not None,
            r["sharpe_ratio"] if r["sharpe_ratio"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    header = f"{'Fast':>6} | {'Slow':>6} | {'Total Return':>13} | {'Sharpe Ratio':>13} | {'Max Drawdown':>13} | {'Win Rate':>10} | {'Trades':>7}"
    print(header)
    print("-" * len(header))

    for item in sorted_results:
        ret_str = _format_metric(item.get("total_return"), is_percentage=True)
        sharpe_str = _format_metric(item.get("sharpe_ratio"))
        mdd_str = _format_metric(item.get("max_drawdown"), is_percentage=True)
        wr_str = _format_metric(item.get("win_rate"), is_percentage=True)
        trades_str = _format_metric(item.get("total_trades"))

        print(
            f"{item['fast_window']:>6} | {item['slow_window']:>6} | "
            f"{ret_str:>13} | {sharpe_str:>13} | {mdd_str:>13} | "
            f"{wr_str:>10} | {trades_str:>7}"
        )

    print("=" * len(header))

    valid_sharpes = [r for r in sorted_results if r.get("sharpe_ratio") is not None]
    if valid_sharpes:
        best = valid_sharpes[0]
        worst = valid_sharpes[-1]
        print(
            f"Best by Sharpe:  Fast={best['fast_window']}, Slow={best['slow_window']} "
            f"(Sharpe={_format_metric(best['sharpe_ratio'])}, Return={_format_metric(best['total_return'], is_percentage=True)})"
        )
        print(
            f"Worst by Sharpe: Fast={worst['fast_window']}, Slow={worst['slow_window']} "
            f"(Sharpe={_format_metric(worst['sharpe_ratio'])}, Return={_format_metric(worst['total_return'], is_percentage=True)})"
        )
        print("=" * len(header))


def main(args: list[str] | None = None) -> list[dict[str, Any]]:
    parsed_args = parse_args(args)
    try:
        df = get_ohlcv(
            symbol=parsed_args.symbol,
            timeframe=parsed_args.timeframe,
            since=parsed_args.since,
            until=parsed_args.until,
        )
        close = df["close"]
        results = run_parameter_sweep(
            close=close,
            fast_windows=parsed_args.fast_windows,
            slow_windows=parsed_args.slow_windows,
            init_cash=parsed_args.init_cash,
            fees=parsed_args.fees,
            slippage=parsed_args.slippage,
        )
        print_report(parsed_args, results)
        return results
    except Exception as e:
        print(f"Error executing parameter sweep demo: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
