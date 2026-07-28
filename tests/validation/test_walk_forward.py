"""Unit tests for walk-forward validation engine."""

import numpy as np
import pandas as pd
import pytest

from quantlab.strategies.sma_crossover import generate_signals
from quantlab.validation.walk_forward import (
    aggregate_walk_forward_results,
    generate_folds,
    run_walk_forward,
)


def test_generate_folds_correct_sequence():
    """Verify generate_folds produces sequential non-overlapping folds and stops before incomplete ones."""
    folds = generate_folds("2018-01-01", "2020-01-01", train_months=9, test_months=3)
    assert len(folds) == 2

    # Fold 1
    assert folds[0]["train_start"] == pd.Timestamp("2018-01-01")
    assert folds[0]["train_end"] == pd.Timestamp("2018-10-01")
    assert folds[0]["test_start"] == pd.Timestamp("2018-10-01")
    assert folds[0]["test_end"] == pd.Timestamp("2019-01-01")

    # Fold 2
    assert folds[1]["train_start"] == pd.Timestamp("2019-01-01")
    assert folds[1]["train_end"] == pd.Timestamp("2019-10-01")
    assert folds[1]["test_start"] == pd.Timestamp("2019-10-01")
    assert folds[1]["test_end"] == pd.Timestamp("2020-01-01")

    # Check incomplete trailing fold handling (18 months pool, 12 months per fold)
    folds_partial = generate_folds("2018-01-01", "2019-06-01", train_months=9, test_months=3)
    assert len(folds_partial) == 1
    assert folds_partial[0]["test_end"] == pd.Timestamp("2019-01-01")


def test_generate_folds_invalid_inputs():
    """Verify generate_folds raises ValueError for invalid input parameters."""
    with pytest.raises(ValueError, match="must be > 0"):
        generate_folds("2018-01-01", "2020-01-01", train_months=0, test_months=3)

    with pytest.raises(ValueError, match="must be > 0"):
        generate_folds("2018-01-01", "2020-01-01", train_months=9, test_months=-1)

    with pytest.raises(ValueError, match="must be strictly earlier"):
        generate_folds("2020-01-01", "2020-01-01", train_months=9, test_months=3)

    with pytest.raises(ValueError, match="must be strictly earlier"):
        generate_folds("2021-01-01", "2020-01-01", train_months=9, test_months=3)


def test_run_walk_forward_synthetic_data():
    """Verify run_walk_forward produces at least one 'ok' fold with valid metrics."""
    # Create 2 years of synthetic price data with oscillations
    dates = pd.date_range("2018-01-01", "2020-01-01", freq="1D", tz="UTC")
    np.random.seed(42)
    # Sine wave with trend to guarantee SMA crossovers
    x = np.linspace(0, 10 * np.pi, len(dates))
    prices = 100.0 + 10.0 * np.sin(x) + np.linspace(0, 50, len(dates))
    close = pd.Series(prices, index=dates)

    folds = generate_folds("2018-01-01", "2020-01-01", train_months=9, test_months=3)
    results = run_walk_forward(
        close=close,
        folds=folds,
        fast_windows=[5, 10],
        slow_windows=[20, 30],
        min_trades=1,
    )

    assert len(results) == len(folds)
    ok_folds = [r for r in results if r["status"] == "ok"]
    assert len(ok_folds) >= 1

    fold = ok_folds[0]
    assert fold["selected_fast"] in [5, 10]
    assert fold["selected_slow"] in [20, 30]
    assert fold["selected_fast"] < fold["selected_slow"]

    oos = fold["oos_metrics"]
    assert isinstance(oos, dict)
    assert "total_return" in oos
    assert "sharpe_ratio" in oos
    assert "max_drawdown" in oos
    assert "win_rate" in oos
    assert "total_trades" in oos
    assert isinstance(oos["total_trades"], int)


def test_run_walk_forward_skipped_fold():
    """Verify fold with no config meeting min_trades is marked skipped_no_valid_config."""
    dates = pd.date_range("2018-01-01", "2020-01-01", freq="1D", tz="UTC")
    close = pd.Series(100.0, index=dates)  # Flat price -> 0 trades

    folds = generate_folds("2018-01-01", "2020-01-01", train_months=9, test_months=3)
    results = run_walk_forward(
        close=close,
        folds=folds,
        fast_windows=[5, 10],
        slow_windows=[20, 30],
        min_trades=5,
    )

    assert len(results) == len(folds)
    for res in results:
        assert res["status"] == "skipped_no_valid_config"
        assert res["selected_fast"] is None
        assert res["selected_slow"] is None
        assert res["train_metrics"] is None
        assert res["oos_metrics"] is None


def test_lookback_buffer_behavior():
    """Verify indicator lookback buffer prevents warm-up NaNs in test window early bars."""
    dates = pd.date_range("2020-01-01", periods=100, freq="1D", tz="UTC")
    prices = np.linspace(100, 200, 100)
    close = pd.Series(prices, index=dates)

    test_start = dates[50]
    test_end = dates[99]

    # Without buffer: slice close directly on test window
    close_no_buf = close[test_start:test_end]
    # Slow SMA = 30 lookback. First 29 bars without buffer will produce NaN signals
    entries_no_buf, _ = generate_signals(close_no_buf, fast_window=5, slow_window=30)
    # The early bars of close_no_buf fail to have valid SMA lookbacks
    # Check that in close_no_buf, SMAs for first 29 bars are NaN
    from quantlab.indicators.trend import sma
    slow_sma_no_buf = sma(close_no_buf, 30)
    assert slow_sma_no_buf.iloc[0:29].isna().all()

    # WITH buffer: slice close including prior lookback days (30 days before test_start)
    buf_start = test_start - pd.Timedelta(days=30)
    close_buf = close[buf_start:test_end]
    slow_sma_buf = sma(close_buf, 30)
    # Extract SMA values aligned with test_start onwards
    slow_sma_at_test_start = slow_sma_buf.loc[close_no_buf.index]
    # In test window, the very first bar is now non-NaN because of the buffer!
    assert not slow_sma_at_test_start.iloc[0:29].isna().any()


def test_aggregate_walk_forward_results():
    """Verify aggregate_walk_forward_results correctly aggregates fold metrics and handles empty case."""
    fold_results = [
        {
            "status": "ok",
            "selected_fast": 5,
            "selected_slow": 20,
            "oos_metrics": {
                "total_return": 0.10,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.05,
                "win_rate": 0.6,
                "total_trades": 5,
            },
        },
        {
            "status": "ok",
            "selected_fast": 10,
            "selected_slow": 30,
            "oos_metrics": {
                "total_return": -0.02,
                "sharpe_ratio": -0.5,
                "max_drawdown": -0.15,
                "win_rate": 0.4,
                "total_trades": 3,
            },
        },
        {
            "status": "skipped_no_valid_config",
            "selected_fast": None,
            "selected_slow": None,
            "oos_metrics": None,
        },
    ]

    agg = aggregate_walk_forward_results(fold_results)
    assert agg["folds_total"] == 3
    assert agg["folds_evaluated"] == 2
    assert agg["folds_skipped"] == 1
    assert pytest.approx(agg["avg_oos_return"]) == 0.04
    assert pytest.approx(agg["avg_oos_sharpe"]) == 0.5
    assert agg["worst_oos_drawdown"] == -0.15
    assert pytest.approx(agg["pct_folds_profitable"]) == 0.5
    assert agg["total_oos_trades_summed"] == 8
    assert agg["selected_configs_per_fold"] == [(5, 20), (10, 30)]

    # Edge case: folds_evaluated == 0
    skipped_results = [
        {
            "status": "skipped_no_valid_config",
            "selected_fast": None,
            "selected_slow": None,
            "oos_metrics": None,
        }
    ]
    agg_zero = aggregate_walk_forward_results(skipped_results)
    assert agg_zero["folds_total"] == 1
    assert agg_zero["folds_evaluated"] == 0
    assert agg_zero["folds_skipped"] == 1
    assert agg_zero["avg_oos_return"] is None
    assert agg_zero["avg_oos_sharpe"] is None
    assert agg_zero["worst_oos_drawdown"] is None
    assert agg_zero["pct_folds_profitable"] is None
    assert agg_zero["total_oos_trades_summed"] == 0
    assert agg_zero["selected_configs_per_fold"] == []
