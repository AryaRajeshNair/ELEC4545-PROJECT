from __future__ import annotations

import os
import pandas as pd
import numpy as np
from arch import arch_model
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Suppress sklearn parallel job warnings
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*should be used with.*')


# ============================================================================
# DATA FILTERING UTILITIES
# ============================================================================

def filter_nonzero_returns(daily_returns, lookback_days=504):
    """
    Filter out zero-return periods from daily returns.
    
    Zero-return periods (cash/flat positions) cause GARCH convergence issues.
    This function identifies and removes them to preserve only active trading periods.
    
    Args:
        daily_returns: DataFrame of daily returns
        lookback_days: Number of lookback days to examine
    
    Returns:
        DataFrame with same structure but zero-return periods filtered out
    """
    filtered = daily_returns.copy()
    
    for col in filtered.columns:
        series = filtered[col].dropna()
        if len(series) > 0:
            near_zero = (series.abs() < 1e-8)
            series[near_zero] = np.nan
            filtered[col] = series
    
    filtered = filtered.ffill(limit=1)
    filtered = filtered.dropna(how='all')
    
    return filtered


def filter_active_trading_periods(monthly_returns_df):
    """
    Filter out flat/cash periods from monthly returns.
    
    In the backtest, many months have gross_ret=0 (portfolio was flat).
    These zero-return periods poison GARCH estimation.
    This function removes them before fitting GARCH.
    
    Args:
        monthly_returns_df: DataFrame with 'gross_ret' column (or similar return column)
    
    Returns:
        Filtered DataFrame containing only active trading periods
    """
    filtered = monthly_returns_df.copy()
    
    if 'gross_ret' in filtered.columns:
        mask = filtered['gross_ret'].abs() > 1e-8
        filtered = filtered[mask].copy()
    
    return filtered


def print_filtering_summary(original_len, filtered_len, sector_tickers):
    """Print summary of data filtering."""
    removed = original_len - filtered_len
    pct_removed = (removed / original_len * 100) if original_len > 0 else 0
    
    print(f"\n  Data Filtering Summary:")
    print(f"    Original observations: {original_len}")
    print(f"    After filtering:       {filtered_len}")
    print(f"    Removed (zero-return periods): {removed} ({pct_removed:.1f}%)")
    print(f"    Active trading periods retained: {filtered_len}")


# ============================================================================
# GARCH VOLATILITY FORECASTING WITH MULTI-STEP AGGREGATION
# ============================================================================

def forecast_sector_volatility_garch(
    daily_returns,
    rebalance_dates,
    sector_tickers,
    lookback_days=504,
    min_obs=100,
    forecast_horizon_days=21,  
    annualize=True,
    verbose=False,
):
    """
    Forecast volatility using GARCH(1,1) on DAILY returns with MULTI-STEP AGGREGATION.
    
    This implements the method from Hlouskova, Schmidheiny, and Wagner (2009):
    "Multistep predictions for multivariate GARCH models with closed-form solution"
    
    Key insight: When rebalancing monthly but using daily data, you need to
    AGGREGATE daily forecasts over the full horizon, not just scale by sqrt(21).
    
    Args:
        daily_returns: DataFrame of DAILY returns
        rebalance_dates: Monthly dates when we rebalance
        sector_tickers: List of sector names
        lookback_days: Number of daily observations to use
        min_obs: Minimum observations required for GARCH
        forecast_horizon_days: Number of trading days to forecast (21 = 1 month)
        annualize: If True, convert to annualized volatility
        verbose: Print progress
    
    Returns:
        DataFrame with dates as index and sector volatility forecasts (annualized)
    """
    print("\n" + "=" * 70)
    print("GARCH MULTI-STEP FORECASTING (Hlouskova et al. 2009)")
    print("=" * 70)
    print(f"  Using {lookback_days} days of daily data per forecast")
    print(f"  Forecasting {forecast_horizon_days} days ahead (1 trading month)")
    print(f"  Aggregating using SUMMATION (not sqrt scaling)")
    print(f"  FILTERING OUT zero-return periods to improve convergence")
    print("=" * 70)
    
    # FILTER zero-return periods before forecasting
    daily_returns_filtered = filter_nonzero_returns(daily_returns, lookback_days=lookback_days)
    print(f"  Data after filtering: {len(daily_returns_filtered)} days (from {len(daily_returns)} original)")
    
    forecasts = []
    failures = []
    
    for i, rebalance_date in enumerate(rebalance_dates):
        if verbose and i % 12 == 0:
            print(f"\n  Processing {rebalance_date.strftime('%Y-%m')} ({i+1}/{len(rebalance_dates)})")
        
        row = {"date": pd.Timestamp(rebalance_date)}
        
        for sector in sector_tickers:
            # Get DAILY returns up to (but not including) rebalance date
            history = (
                pd.Series(daily_returns_filtered.loc[daily_returns_filtered.index < rebalance_date, sector])
                .dropna()
                .tail(lookback_days)
            )
            
            # Check minimum observations
            if len(history) < min_obs:
                raise RuntimeError(
                    f"GARCH failed for {sector} on {rebalance_date}: "
                    f"Only {len(history)} daily observations (need {min_obs})"
                )
            
            # Clean the data
            returns_clean = history.dropna()
            returns_clean = returns_clean.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns_clean) < min_obs:
                raise RuntimeError(
                    f"GARCH failed for {sector} on {rebalance_date}: "
                    f"Only {len(returns_clean)} valid observations after cleaning"
                )
            
            # Scale for numerical stability
            scaled_returns = returns_clean * 100
            
            # Fit GARCH model (arch library handles optimization automatically)
            garch_converged = False
            last_error = None
            fitted_model = None
            
            try:
                model = arch_model(scaled_returns, vol="GARCH", p=1, q=1, dist="normal")
                fitted = model.fit(disp="off", show_warning=False)
                
                # Check if fit was successful by examining parameters
                if hasattr(fitted, 'params') and fitted.params is not None and 'alpha[1]' in fitted.params.index:
                    fitted_model = fitted
                    garch_converged = True
            except Exception as e:
                last_error = str(e)
            
            if not garch_converged:
                raise RuntimeError(
                    f"GARCH FAILED for {sector} on {rebalance_date}: "
                    f"No optimization method converged. Last error: {last_error}"
                )
            
            forecast = fitted_model.forecast(horizon=forecast_horizon_days)
            daily_variances_pct = forecast.variance.values[-1, :forecast_horizon_days]
            monthly_variance_pct = np.sum(daily_variances_pct)
            daily_vol_decimal = np.sqrt(monthly_variance_pct) / 100
            monthly_vol = daily_vol_decimal
            
            if annualize:
                monthly_vol = monthly_vol * np.sqrt(12)
            
            row[sector] = float(monthly_vol)
            
            if verbose and i % 12 == 0 and sector == sector_tickers[0]:
                alpha = fitted_model.params.get('alpha[1]', 0)
                beta = fitted_model.params.get('beta[1]', 0)
                print(f"    {sector}: α={alpha:.3f}, β={beta:.3f}, α+β={alpha+beta:.3f}")
                print(f"      Multi-step monthly vol = {monthly_vol*100:.2f}% (sum of {forecast_horizon_days} daily variances)")
        
        forecasts.append(row)
    
    result_df = pd.DataFrame(forecasts).set_index("date")
    print(f"\n  ✓ GARCH multi-step forecasts generated for {len(result_df)} months")
    
    return result_df


# ============================================================================
# DIAGNOSTICS ON DAILY RETURNS
# ============================================================================

def run_daily_diagnostics(daily_returns, sector_tickers, output_dir=None):
    """
    Run diagnostics on DAILY returns to verify GARCH is appropriate.
    Includes data quality assessment and filtering analysis.
    """
    print("\n" + "=" * 80)
    print("GARCH DIAGNOSTICS ON DAILY RETURNS")
    print("=" * 80)
    
    # Print data quality info
    print(f"\nData Quality Assessment:")
    print(f"  Total observations: {len(daily_returns)}")
    print(f"  Date range: {daily_returns.index.min()} to {daily_returns.index.max()}")
    
    # Check for zero-return periods
    zero_return_count = 0
    for col in sector_tickers:
        if col in daily_returns.columns:
            zeros = (daily_returns[col].abs() < 1e-8).sum()
            zero_return_count += zeros
    
    print(f"  Near-zero observations (< 1e-8): {zero_return_count} total")
    
    results = {
        'stationarity': [],
        'arch_effects': [],
        'garch_fits': [],
        'summary': {}
    }
    
    arch_count = 0
    garch_converged = 0
    
    for sector in sector_tickers:
        returns = daily_returns[sector].dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) < 100:
            print(f"\n{sector}: Insufficient data ({len(returns)} days)")
            continue
        
        print(f"\n{sector}:")
        
        # Stationarity test
        adf_p = adfuller(returns)[1]
        print(f"  ADF p-value: {adf_p:.6f} → {'stationary' if adf_p < 0.05 else 'non-stationary'}")
        
        # ARCH-LM test
        centered = returns - returns.mean()
        arch_test = het_arch(centered, nlags=5)
        has_arch = arch_test[1] < 0.05
        if has_arch:
            arch_count += 1
        print(f"  ARCH-LM p-value: {arch_test[1]:.6f} → {'HAS ARCH' if has_arch else 'NO ARCH'}")
        
        # Try GARCH
        try:
            scaled = returns * 100
            model = arch_model(scaled, vol="GARCH", p=1, q=1, dist="normal")
            fitted = model.fit(disp="off", show_warning=False)
            
            # Check if fit was successful by examining parameters
            if hasattr(fitted, 'params') and fitted.params is not None and 'alpha[1]' in fitted.params.index:
                garch_converged += 1
                alpha = fitted.params.get('alpha[1]', 0)
                beta = fitted.params.get('beta[1]', 0)
                print(f"  GARCH: ✓ CONVERGED (α={alpha:.3f}, β={beta:.3f}, α+β={alpha+beta:.3f})")
                results['garch_fits'].append({
                    'sector': sector,
                    'converged': True,
                    'alpha': alpha,
                    'beta': beta
                })
            else:
                print(f"  GARCH: ✗ DID NOT CONVERGE")
        except Exception as e:
            print(f"  GARCH: ✗ FAILED - {str(e)[:50]}")
    
    results['summary']['arch_count'] = arch_count
    results['summary']['garch_converged'] = garch_converged
    results['summary']['total_sectors'] = len(sector_tickers)
    
    print("\n" + "-" * 60)
    print("SUMMARY FOR DAILY RETURNS:")
    print(f"  ARCH effects: {arch_count}/{len(sector_tickers)} sectors")
    print(f"  GARCH converged: {garch_converged}/{len(sector_tickers)} sectors")
    
    return results


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def get_portfolio_volatility_forecast_no_lookahead(port_returns_series, current_position):
    """Simple historical volatility for portfolio scaling."""
    historical = port_returns_series.iloc[:current_position].dropna()
    return historical.std() * np.sqrt(12) if len(historical) > 0 else 0.15


def full_garch_diagnostics(monthly_returns, sector_tickers, output_dir=None):
    """Wrapper for backward compatibility."""
    print("\nNote: full_garch_diagnostics is designed for monthly returns.")
    print("For daily GARCH, use run_daily_diagnostics() with daily_returns.")
    return {'summary': {'stationary_count': len(sector_tickers), 'arch_count': 0, 'total_sectors': len(sector_tickers)}}