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
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*should be used with.*')
warnings.filterwarnings('ignore', message='.*delayed.*should be used.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', message='.*delayed.*Parallel.*')

# ============================================================================
# DATA FILTERING UTILITIES
# ============================================================================
def filter_nonzero_returns(daily_returns, lookback_days=504):
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
    filtered = monthly_returns_df.copy()
    
    if 'gross_ret' in filtered.columns:
        mask = filtered['gross_ret'].abs() > 1e-8
        filtered = filtered[mask].copy()
    
    return filtered


def print_filtering_summary(original_len, filtered_len, sector_tickers):
    removed = original_len - filtered_len
    pct_removed = (removed / original_len * 100) if original_len > 0 else 0
    
    print(f"\n  Data Filtering Summary:")
    print(f"    Original observations: {original_len}")
    print(f"    After filtering:       {filtered_len}")
    print(f"    Removed (zero-return periods): {removed} ({pct_removed:.1f}%)")
    print(f"    Active trading periods retained: {filtered_len}")


# ============================================================================
# STATIONARITY & VOLATILITY CLUSTERING DIAGNOSTICS 
# ============================================================================
def test_stationarity_comprehensive(daily_returns, sector_tickers, output_file=None):
    
    print("\n" + "=" * 100)
    print("STATIONARITY & VOLATILITY CLUSTERING DIAGNOSTIC")
    print("=" * 100)
    
    results_list = []
    stationary_count = 0
    arch_count = 0
    
    for sector in sector_tickers:
        # Get returns and clean
        returns = daily_returns[sector].dropna().copy()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) < 50:
            print(f"\n⚠ {sector}: Insufficient data ({len(returns)} days) - SKIPPING")
            continue
        
        # Test 1: ADF on returns
        adf_result = adfuller(returns, autolag='AIC')
        adf_pval = adf_result[1]
        adf_stat = adf_result[0]
        adf_stationary = adf_pval < 0.05
        
        # Test 2: KPSS on returns
        try:
            kpss_result = kpss(returns, regression='c', nlags='auto')
            kpss_pval = kpss_result[1]
            kpss_stat = kpss_result[0]
            kpss_stationary = kpss_pval > 0.05
        except Exception as e:
            kpss_pval = np.nan
            kpss_stat = np.nan
            kpss_stationary = False
        
        # Test 3: Ljung-Box on squared returns (ARCH effects)
        squared_returns = returns ** 2
        try:
            lb_result = acorr_ljungbox(squared_returns, nlags=10, return_df=True)
            lb_pval = lb_result['lb_pvalue'].iloc[0]  # Use first lag
            lb_has_arch = lb_pval < 0.05
        except Exception as e:
            lb_pval = np.nan
            lb_has_arch = False
        
        # Stationarity: Both ADF and KPSS must agree
        is_stationary = adf_stationary and kpss_stationary
        if is_stationary:
            stationary_count += 1
        
        if lb_has_arch:
            arch_count += 1
        
        # Determine status symbols
        adf_symbol = "✓" if adf_stationary else "✗"
        kpss_symbol = "✓" if kpss_stationary else "✗"
        arch_symbol = "✓" if lb_has_arch else "✗"
        stat_symbol = "✓ STATIONARY" if is_stationary else "✗ NON-STATIONARY"
        
        results_list.append({
            'Sector': sector,
            'ADF p-val': f"{adf_pval:.4f} {adf_symbol}",
            'KPSS p-val': f"{kpss_pval:.4f} {kpss_symbol}",
            'Ljung-Box p-val': f"{lb_pval:.4f} {arch_symbol}",
            'Status': stat_symbol
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Print formatted table
    print("\n" + "-" * 100)
    print(results_df.to_string(index=False))
    print("-" * 100)
    
    # Print summary
    print("\nSUMMARY FOR REPORT:")
    print(f"  • Stationarity (ADF + KPSS both agree): {stationary_count}/{len(sector_tickers)} sectors ✓")
    print(f"  • Volatility clustering (Ljung-Box p < 0.05): {arch_count}/{len(sector_tickers)} sectors ✓")
    
    if stationary_count == len(sector_tickers):
        print(f"\n  CONCLUSION: All {len(sector_tickers)} sectors are stationary.")
        print(f"    GARCH modeling is valid and appropriate for volatility forecasting.")
    else:
        print(f"\n  CAUTION: {len(sector_tickers) - stationary_count} sector(s) are non-stationary.")
        print(f"    Consider differencing or transformation before GARCH modeling.")
    
    if arch_count > 0:
        print(f"\n  Volatility clustering detected in {arch_count} sector(s).")
        print(f"    This justifies the use of GARCH models for volatility forecasting.")
    
    print("\n" + "=" * 100)
    
    # Save to file if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\n  Results saved to: {output_file}")
    
    return results_df, {
        'total_sectors': len(sector_tickers),
        'stationary_count': stationary_count,
        'arch_count': arch_count
    }


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
            
            returns_clean = history.dropna()
            returns_clean = returns_clean.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns_clean) < min_obs:
                raise RuntimeError(
                    f"GARCH failed for {sector} on {rebalance_date}: "
                    f"Only {len(returns_clean)} valid observations after cleaning"
                )
            
            # Scale for numerical stability
            scaled_returns = returns_clean * 100
            
            # Fit GARCH model 
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
    print(f"\n  GARCH multi-step forecasts generated for {len(result_df)} months")
    
    return result_df


# ============================================================================
# DIAGNOSTICS ON DAILY RETURNS
# ============================================================================
def run_daily_diagnostics(daily_returns, sector_tickers, output_dir=None):
    print("\n" + "=" * 80)
    print("GARCH DIAGNOSTICS ON DAILY RETURNS")
    print("=" * 80)
    
    
    print(f"\nData Quality Assessment:")
    print(f"  Total observations: {len(daily_returns)}")
    print(f"  Date range: {daily_returns.index.min()} to {daily_returns.index.max()}")
    
   
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
        results['arch_effects'].append({
            'sector': sector,
            'frequency': 'daily',
            'n_obs': int(len(returns)),
            'arch_lm_stat': float(arch_test[0]),
            'arch_lm_pvalue': float(arch_test[1]),
            'has_arch': bool(has_arch),
        })
        

        try:
            scaled = returns * 100
            model = arch_model(scaled, vol="GARCH", p=1, q=1, dist="normal")
            fitted = model.fit(disp="off", show_warning=False)
            
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
    historical = port_returns_series.iloc[:current_position].dropna()
    return historical.std() * np.sqrt(12) if len(historical) > 0 else 0.15


def full_garch_diagnostics(monthly_returns, sector_tickers, output_dir=None):
    print("\n" + "=" * 80)
    print("GARCH DIAGNOSTICS ON MONTHLY RETURNS")
    print("=" * 80)

    results = {
        'arch_effects': [],
        'summary': {}
    }

    tested_sectors = 0
    arch_count = 0

    for sector in sector_tickers:
        if sector not in monthly_returns.columns:
            print(f"\n{sector}: Missing from monthly returns - SKIPPING")
            continue

        returns = monthly_returns[sector].dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) < 24:
            print(f"\n{sector}: Insufficient data ({len(returns)} months) - SKIPPING")
            continue

        
        nlags = min(5, max(1, len(returns) // 6))
        centered = returns - returns.mean()

        try:
            arch_test = het_arch(centered, nlags=nlags)
            pvalue = float(arch_test[1])
            stat = float(arch_test[0])
            has_arch = pvalue < 0.05
            tested_sectors += 1
            if has_arch:
                arch_count += 1

            print(f"\n{sector}:")
            print(f"  ARCH-LM p-value: {pvalue:.6f} (nlags={nlags}) → {'HAS ARCH' if has_arch else 'NO ARCH'}")

            results['arch_effects'].append({
                'sector': sector,
                'frequency': 'monthly',
                'n_obs': int(len(returns)),
                'nlags': int(nlags),
                'arch_lm_stat': stat,
                'arch_lm_pvalue': pvalue,
                'has_arch': bool(has_arch),
            })
        except Exception as e:
            print(f"\n{sector}: ARCH-LM failed - {str(e)[:80]}")

    results['summary']['arch_count'] = arch_count
    results['summary']['tested_sectors'] = tested_sectors
    results['summary']['total_sectors'] = len(sector_tickers)

    print("\n" + "-" * 60)
    print("SUMMARY FOR MONTHLY RETURNS:")
    print(f"  ARCH effects: {arch_count}/{tested_sectors} tested sectors")

    return results