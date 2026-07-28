"""Run Walk-Forward Validation Demo on historical Binance OHLCV data."""

import argparse
import sys
from typing import Any
import pandas as pd

from quantlab.data.ohlcv import get_ohlcv
from quantlab.validation.walk_forward import (
    aggregate_walk_forward_results,
    generate_folds,
    run_walk_forward,
)


def resolve_window_list(
    windows: list[int] | None,
    start: int | None,
    stop: int | None,
    step: int | None,
    default_list: list[int],
) -> list[int]:
    """Resolve a list of window lookbacks from explicit list or range parameters.

    Args:
        windows: Optional list of integer windows.
        start: Optional range start.
        stop: Optional range stop (inclusive).
        step: Optional range step.
        default_list: Fallback default window list.

    Returns:
        list[int]: Resolved list of window integers.
    """
    if start is not None and stop is not None:
        st = step if step is not None else 1
        return list(range(start, stop + 1, st))
    if windows is not None and len(windows) > 0:
        return windows
    return default_list


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Walk-Forward Strategy Validation on historical Binance data."
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
        "--pool-start",
        type=str,
        required=True,
        help="Start date of data pool (e.g. 2018-01-01)",
    )
    parser.add_argument(
        "--pool-end",
        type=str,
        required=True,
        help="End date of data pool (e.g. 2024-01-01)",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=9,
        help="Training window duration in months (default: 9)",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=3,
        help="Testing window duration in months (default: 3)",
    )
    parser.add_argument(
        "--fast-windows",
        type=int,
        nargs="+",
        default=None,
        help="Fast SMA window lookbacks (e.g. 5 10 20)",
    )
    parser.add_argument(
        "--fast-start",
        type=int,
        default=None,
        help="Fast SMA window range start",
    )
    parser.add_argument(
        "--fast-stop",
        type=int,
        default=None,
        help="Fast SMA window range stop (inclusive)",
    )
    parser.add_argument(
        "--fast-step",
        type=int,
        default=None,
        help="Fast SMA window range step",
    )
    parser.add_argument(
        "--slow-windows",
        type=int,
        nargs="+",
        default=None,
        help="Slow SMA window lookbacks (e.g. 20 50 100)",
    )
    parser.add_argument(
        "--slow-start",
        type=int,
        default=None,
        help="Slow SMA window range start",
    )
    parser.add_argument(
        "--slow-stop",
        type=int,
        default=None,
        help="Slow SMA window range stop (inclusive)",
    )
    parser.add_argument(
        "--slow-step",
        type=int,
        default=None,
        help="Slow SMA window range step",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum trades required in train window (default: 10)",
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


def print_walk_forward_report(
    args: argparse.Namespace,
    folds: list[dict[str, pd.Timestamp]],
    fold_results: list[dict[str, Any]],
    agg: dict[str, Any],
) -> None:
    print("=" * 110)
    print("                    WALK-FORWARD VALIDATION REPORT                   ")
    print("=" * 110)
    print(f"Symbol:         {args.symbol}")
    print(f"Timeframe:      {args.timeframe}")
    print(f"Data Pool:      {args.pool_start} to {args.pool_end}")
    print(f"Train / Test:   {args.train_months} months train / {args.test_months} months test")
    print(f"Min Trades:     {args.min_trades}")
    print(f"Initial Cash:   ${args.init_cash:,.2f}")
    print(f"Fees / Slip:    {args.fees * 100:.2f}% / {args.slippage * 100:.2f}%")
    print("-" * 110)

    pool_end_ts = pd.Timestamp(args.pool_end)
    if folds:
        last_fold_end = folds[-1]["test_end"]
        unused_days = (pool_end_ts - last_fold_end).days
    else:
        unused_days = (pool_end_ts - pd.Timestamp(args.pool_start)).days

    print(
        f"Generated {len(folds)} fold(s). Unused time at pool end: {unused_days} days "
        f"({unused_days / 30.4375:.1f} months)."
    )
    print("-" * 110)

    # Per-fold table
    header = (
        f"{'Fold':>4} | {'Train Range':>23} | {'Test Range':>23} | "
        f"{'Status':>23} | {'Config':>9} | {'OOS Return':>11} | {'OOS Sharpe':>10} | {'OOS Drawdown':>12} | {'OOS Trades':>10}"
    )
    print(header)
    print("-" * len(header))

    for i, res in enumerate(fold_results, start=1):
        tr_start_str = res["train_start"].strftime("%Y-%m-%d")
        tr_end_str = res["train_end"].strftime("%Y-%m-%d")
        te_start_str = res["test_start"].strftime("%Y-%m-%d")
        te_end_str = res["test_end"].strftime("%Y-%m-%d")

        train_range = f"{tr_start_str}..{tr_end_str}"
        test_range = f"{te_start_str}..{te_end_str}"

        status = res["status"]
        if status == "ok":
            config_str = f"({res['selected_fast']}, {res['selected_slow']})"
            oos = res["oos_metrics"]
            ret_str = _format_metric(oos.get("total_return"), is_percentage=True)
            sharpe_str = _format_metric(oos.get("sharpe_ratio"))
            mdd_str = _format_metric(oos.get("max_drawdown"), is_percentage=True)
            trades_str = _format_metric(oos.get("total_trades"))
        else:
            config_str = "None"
            ret_str = "N/A"
            sharpe_str = "N/A"
            mdd_str = "N/A"
            trades_str = "N/A"

        print(
            f"{i:>4} | {train_range:>23} | {test_range:>23} | "
            f"{status:>23} | {config_str:>9} | {ret_str:>11} | {sharpe_str:>10} | {mdd_str:>12} | {trades_str:>10}"
        )

    print("=" * len(header))
    print("                        WALK-FORWARD AGGREGATE SUMMARY               ")
    print("=" * len(header))
    print(f"Total Folds:              {agg['folds_total']}")
    print(f"Evaluated Folds:          {agg['folds_evaluated']}")
    print(f"Skipped Folds:            {agg['folds_skipped']}")

    avg_ret_str = _format_metric(agg.get("avg_oos_return"), is_percentage=True)
    avg_sharpe_str = _format_metric(agg.get("avg_oos_sharpe"))
    worst_mdd_str = _format_metric(agg.get("worst_oos_drawdown"), is_percentage=True)
    pct_prof_str = _format_metric(agg.get("pct_folds_profitable"), is_percentage=True)

    print(f"Avg OOS Return:           {avg_ret_str}")
    print(f"Avg OOS Sharpe Ratio:     {avg_sharpe_str}")
    print(f"Worst OOS Max Drawdown:   {worst_mdd_str}")
    print(f"Profitable Folds %:       {pct_prof_str}")
    print(f"Total OOS Trades Summed:  {agg['total_oos_trades_summed']}")
    print(f"Selected Configs / Fold:  {agg['selected_configs_per_fold']}")
    print("=" * len(header))


def main(args: list[str] | None = None) -> dict[str, Any]:
    parsed_args = parse_args(args)

    fast_windows = resolve_window_list(
        windows=parsed_args.fast_windows,
        start=parsed_args.fast_start,
        stop=parsed_args.fast_stop,
        step=parsed_args.fast_step,
        default_list=[5, 10, 20, 50],
    )
    slow_windows = resolve_window_list(
        windows=parsed_args.slow_windows,
        start=parsed_args.slow_start,
        stop=parsed_args.slow_stop,
        step=parsed_args.slow_step,
        default_list=[20, 50, 100, 200],
    )

    try:
        df = get_ohlcv(
            symbol=parsed_args.symbol,
            timeframe=parsed_args.timeframe,
            since=parsed_args.pool_start,
            until=parsed_args.pool_end,
        )
        close = df["close"]

        folds = generate_folds(
            pool_start=parsed_args.pool_start,
            pool_end=parsed_args.pool_end,
            train_months=parsed_args.train_months,
            test_months=parsed_args.test_months,
        )

        fold_results = run_walk_forward(
            close=close,
            folds=folds,
            fast_windows=fast_windows,
            slow_windows=slow_windows,
            min_trades=parsed_args.min_trades,
            init_cash=parsed_args.init_cash,
            fees=parsed_args.fees,
            slippage=parsed_args.slippage,
        )

        agg = aggregate_walk_forward_results(fold_results)
        print_walk_forward_report(parsed_args, folds, fold_results, agg)
        return agg

    except Exception as e:
        print(f"Error executing walk-forward validation demo: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
