from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*delayed.*should be used.*')
warnings.filterwarnings('ignore', message='.*should be used with.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')

import json
import pandas as pd
import numpy as np

from simple_strategy.config import (
    ALLOW_SHORT,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
    NET_EXPOSURE,
    OPTIMIZER_RISK_AVERSION,
    RESULTS_DIR,
    SECTOR_TICKERS,
    START_DATE,
    TEST_START,
    TRAIN_WINDOW_MONTHS,
    FIGURES_DIR,
)
from simple_strategy.data_prep import create_features, download_market_data
from simple_strategy.garch import (
    forecast_sector_volatility_garch,
    run_daily_diagnostics,
)
from simple_strategy.ml_model import (
    calculate_average_feature_importance,
    evaluate_predictions,
    run_rolling_predictions,
    test_prediction_value,
)
from simple_strategy.portfolio import run_forecast_driven_backtest, run_simple_momentum_backtest
from simple_strategy.figures import generate_all_figures


def save_outputs(
    forecast_backtest,
    baseline_backtest,
    predictions,
    sector_vol_forecasts,
    prediction_metrics,
    sorting_metrics,
    feature_importance,
    garch_diagnostics,
):
    """Save all backtest outputs to CSV and JSON files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    forecast_backtest["trade_log_df"].to_csv(RESULTS_DIR / "trade_log.csv")

    monthly_results = pd.DataFrame(
        {
            "strategy_return": forecast_backtest["returns_series"],
            "strategy_equity": forecast_backtest["equity_series"],
            "baseline_return": baseline_backtest["returns_series"].reindex(forecast_backtest["returns_series"].index),
        }
    )
    monthly_results.to_csv(RESULTS_DIR / "backtest_monthly_results.csv")

    equity_curves = pd.DataFrame(
        {
            "strategy_equity": forecast_backtest["equity_series"],
            "baseline_equity": baseline_backtest["equity_series"].reindex(forecast_backtest["equity_series"].index),
        }
    )
    equity_curves.to_csv(RESULTS_DIR / "equity_curves.csv")

    predictions.to_csv(RESULTS_DIR / "rf_predictions.csv")
    sector_vol_forecasts.to_csv(RESULTS_DIR / "sector_vol_forecasts.csv")

    if feature_importance is not None:
        feature_importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    if garch_diagnostics:
        garch_summary = pd.DataFrame([garch_diagnostics['summary']])
        garch_summary.to_csv(RESULTS_DIR / "garch_diagnostics_summary.csv", index=False)

    metrics = {
        "strategy_sharpe": float(forecast_backtest["sharpe"]),
        "strategy_max_drawdown": float(forecast_backtest["max_drawdown"]),
        "strategy_win_rate": float(forecast_backtest["win_rate"]),
        "strategy_total_return_pct": float(forecast_backtest["total_return"] * 100),
        "baseline_sharpe": float(baseline_backtest["sharpe"]),
        "baseline_max_drawdown": float(baseline_backtest["max_drawdown"]),
        "baseline_win_rate": float(baseline_backtest["win_rate"]),
        "baseline_total_return_pct": float(baseline_backtest["total_return"] * 100),
        "directional_accuracy": float(prediction_metrics["directional_accuracy"]),
        "prediction_correlation": float(prediction_metrics["correlation"]),
        "prediction_sorting_spread_pct": float(sorting_metrics[0] * 100) if sorting_metrics[0] == sorting_metrics[0] else None,
        "prediction_sorting_p_value": float(sorting_metrics[1]) if sorting_metrics[1] == sorting_metrics[1] else None,
    }

    with open(RESULTS_DIR / "backtest_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    summary_rows = [
        {"Metric": "Strategy Sharpe", "Value": metrics["strategy_sharpe"]},
        {"Metric": "Strategy Max Drawdown", "Value": metrics["strategy_max_drawdown"]},
        {"Metric": "Strategy Win Rate", "Value": metrics["strategy_win_rate"]},
        {"Metric": "Strategy Total Return %", "Value": metrics["strategy_total_return_pct"]},
        {"Metric": "Baseline Sharpe", "Value": metrics["baseline_sharpe"]},
        {"Metric": "Directional Accuracy", "Value": metrics["directional_accuracy"]},
        {"Metric": "Prediction Correlation", "Value": metrics["prediction_correlation"]},
        {"Metric": "Prediction Sorting Spread %", "Value": metrics["prediction_sorting_spread_pct"]},
        {"Metric": "Prediction Sorting p-value", "Value": metrics["prediction_sorting_p_value"]},
    ]
    
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "backtest_summary_table.csv", index=False)
    
    print(f"\n✓ Results saved to {RESULTS_DIR}")


def main():
    print("\n" + "=" * 80)
    print("FORECAST-DRIVEN PORTFOLIO OPTIMIZATION WITH GARCH (DAILY RETURNS)")
    print("=" * 80)

    # Step 1: Download market data
    print("\n[STEP 1] Downloading market data...")
    market_data = download_market_data(sector_tickers=SECTOR_TICKERS, start_date=START_DATE)
    monthly_returns = market_data["monthly_returns"].dropna(how="all")
    monthly_close = market_data["monthly_close"]
    
    # Get daily returns for GARCH estimation
    daily_returns = market_data["daily_close"].pct_change().dropna(how="all")
    
    if len(monthly_returns) == 0:
        print("ERROR: No data downloaded. Check your internet connection and tickers.")
        return
    
    # Step 1b: Run diagnostics on DAILY returns
    print("\n[STEP 1b] Running GARCH diagnostics on DAILY returns...")
    daily_diagnostics = run_daily_diagnostics(daily_returns, SECTOR_TICKERS, output_dir=FIGURES_DIR)
    
    # Step 2: Create features
    print("\n[STEP 2] Creating features with technical indicators...")
    features, vix_norm, tnx_norm = create_features(
        monthly_close,
        monthly_returns,
        market_data["monthly_vix"],
        market_data["monthly_tnx"],
        min_periods=12,
    )

    # Step 3: Generate predictions
    print("\n[STEP 3] Training Random Forest and generating predictions...")
    predictions = run_rolling_predictions(
        features,
        monthly_returns,
        vix_norm,
        tnx_norm,
        sector_tickers=SECTOR_TICKERS,
        test_start=TEST_START,
        train_window_months=TRAIN_WINDOW_MONTHS,
        use_hyperparameter_tuning=True,
    )

    if len(predictions) == 0:
        print("ERROR: No predictions generated.")
        return

    # Step 3b: Feature importance
    print("\n[STEP 3b] Calculating feature importance...")
    try:
        feature_importance = calculate_average_feature_importance(
            predictions, features, monthly_returns, vix_norm, tnx_norm, SECTOR_TICKERS
        )
    except Exception as e:
        print(f"Warning: Feature importance calculation failed: {e}")
        feature_importance = None

    # Step 4: Evaluate predictions
    print("\n[STEP 4] Evaluating predictions...")
    actual_returns = monthly_returns.reindex(predictions.index)
    prediction_metrics = evaluate_predictions(predictions, actual_returns, sector_tickers=SECTOR_TICKERS)
    sorting_metrics = test_prediction_value(predictions, actual_returns)

    print("\n[STEP 5] Forecasting sector volatilities with GARCH multi-step aggregation...")
    sector_vol_forecasts = forecast_sector_volatility_garch(
        daily_returns,
        predictions.index,
        SECTOR_TICKERS,
        lookback_days=504,
        min_obs=100,
        forecast_horizon_days=21,  
        annualize=True,
        verbose=True,
    )

    # Step 6: Run backtest
    print("\n[STEP 6] Running forecast-driven backtest...")
    forecast_backtest = run_forecast_driven_backtest(
        predictions,
        actual_returns,
        monthly_returns,
        sector_vol_forecasts,
        sector_tickers=SECTOR_TICKERS,
        max_weight=DEFAULT_MAX_WEIGHT,
        min_weight=DEFAULT_MIN_WEIGHT,
        allow_short=ALLOW_SHORT,
        net_exposure=NET_EXPOSURE,
        risk_aversion=OPTIMIZER_RISK_AVERSION,
    )

    # Step 7: Baseline
    print("\n[STEP 7] Running baseline backtest...")
    baseline_backtest = run_simple_momentum_backtest(
        actual_returns,
        monthly_returns,
        sector_tickers=SECTOR_TICKERS,
        use_vol_scaling=False,
    )

    # Step 8: Save outputs
    print("\n[STEP 8] Saving outputs...")
    save_outputs(
        forecast_backtest,
        baseline_backtest,
        predictions,
        sector_vol_forecasts,
        prediction_metrics,
        sorting_metrics,
        feature_importance,
        daily_diagnostics,
    )

    # Step 9: Generate figures
    print("\n[STEP 9] Generating figures...")
    try:
        monthly_results_df = pd.read_csv(
            RESULTS_DIR / "backtest_monthly_results.csv",
            index_col=0,
            parse_dates=True
        )
        generate_all_figures(
            forecast_backtest=forecast_backtest,
            baseline_backtest=baseline_backtest,
            predictions=predictions,
            actual_returns=actual_returns,
            sector_vol_forecasts=sector_vol_forecasts,
            feature_importance=feature_importance,
            monthly_results_df=monthly_results_df,
            figures_dir=FIGURES_DIR,
        )
        print(f"✓ Figures saved to {FIGURES_DIR}")
    except Exception as e:
        print(f"⚠ Figure generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Final results
    print("\n" + "=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"\n{'=' * 40}")
    print("STRATEGY PERFORMANCE")
    print(f"{'=' * 40}")
    print(f"Strategy (PURE GARCH on DAILY returns):")
    print(f"  Sharpe Ratio: {forecast_backtest['sharpe']:.3f}")
    print(f"  Max Drawdown: {forecast_backtest['max_drawdown']:.2%}")
    print(f"  Win Rate: {forecast_backtest['win_rate']:.2%}")
    print(f"  Total Return: {forecast_backtest['total_return']:.2%}")
    
    print(f"\nBaseline (simple momentum):")
    print(f"  Sharpe Ratio: {baseline_backtest['sharpe']:.3f}")
    print(f"  Max Drawdown: {baseline_backtest['max_drawdown']:.2%}")
    print(f"  Win Rate: {baseline_backtest['win_rate']:.2%}")
    print(f"  Total Return: {baseline_backtest['total_return']:.2%}")
    
    print(f"\n{'=' * 40}")
    print("PREDICTION QUALITY")
    print(f"{'=' * 40}")
    print(f"Directional Accuracy: {prediction_metrics['directional_accuracy']:.2%}")
    print(f"Correlation: {prediction_metrics['correlation']:.4f}")
    
    print("\n" + "=" * 80)
    print("All results saved to 'results/' directory")
    print("=" * 80)


if __name__ == "__main__":
    main()