from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (
    ALLOW_SHORT,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
    DEFAULT_TARGET_VOL,
    DEFAULT_TX_COST,
    DEFAULT_LOOKBACK_MONTHS,
    DEFAULT_MIN_OBS,
    NET_EXPOSURE,
    OPTIMIZER_RISK_AVERSION,
    SECTOR_TICKERS,
)
from garch import get_portfolio_volatility_forecast_no_lookahead


def equal_weights(index_like):
    """Return equal weights for all assets."""
    n_assets = len(index_like)
    if n_assets == 0:
        return pd.Series(dtype=float)
    return pd.Series(np.repeat(1.0 / n_assets, n_assets), index=index_like)


def estimate_covariance_matrix(returns_df, current_date, lookback_months=DEFAULT_LOOKBACK_MONTHS, min_obs=DEFAULT_MIN_OBS):
    """Estimate covariance matrix from historical returns."""
    current_date = pd.Timestamp(current_date).normalize()
    end_date = current_date - pd.DateOffset(months=1)
    start_date = end_date - pd.DateOffset(months=lookback_months)
    window = returns_df.loc[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    if len(window) < min_obs:
        window = returns_df.loc[returns_df.index < current_date].tail(min_obs)
    
    if len(window) < 2:
        return pd.DataFrame(np.eye(len(SECTOR_TICKERS)), index=SECTOR_TICKERS, columns=SECTOR_TICKERS)

    cov = window.cov()
    if cov.isna().all().all():
        cov = pd.DataFrame(np.eye(len(SECTOR_TICKERS)), index=SECTOR_TICKERS, columns=SECTOR_TICKERS)
    cov = cov.reindex(index=SECTOR_TICKERS, columns=SECTOR_TICKERS).fillna(0.0)
    
    # FIX: Convert to numpy array and back to avoid read-only issue
    cov_values = cov.values.copy()
    cov_values = cov_values + np.eye(len(SECTOR_TICKERS)) * 1e-6
    cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)
    return cov

def estimate_correlation_matrix(returns_df, current_date, lookback_months=DEFAULT_LOOKBACK_MONTHS, min_obs=DEFAULT_MIN_OBS):
    """Estimate correlation matrix from historical returns."""
    current_date = pd.Timestamp(current_date).normalize()
    end_date = current_date - pd.DateOffset(months=1)
    start_date = end_date - pd.DateOffset(months=lookback_months)
    window = returns_df.loc[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    if len(window) < min_obs:
        window = returns_df.loc[returns_df.index < current_date].tail(min_obs)
    
    if len(window) < 2:
        return pd.DataFrame(np.eye(len(SECTOR_TICKERS)), index=SECTOR_TICKERS, columns=SECTOR_TICKERS)

    corr = window.corr()
    if corr.isna().all().all():
        corr = pd.DataFrame(np.eye(len(SECTOR_TICKERS)), index=SECTOR_TICKERS, columns=SECTOR_TICKERS)
    corr = corr.reindex(index=SECTOR_TICKERS, columns=SECTOR_TICKERS).fillna(0.0)
    
   
    corr_values = corr.values.copy()
    np.fill_diagonal(corr_values, 1.0)
    corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)
    return corr


def build_covariance_from_forecasts(returns_df, current_date, forecast_vols, lookback_months=DEFAULT_LOOKBACK_MONTHS, min_obs=DEFAULT_MIN_OBS):
    """Build covariance matrix using forecasted volatilities and historical correlations."""
    corr = estimate_correlation_matrix(returns_df, current_date, lookback_months=lookback_months, min_obs=min_obs)
    vols = forecast_vols.reindex(corr.index).fillna(0.0)
    
    if (vols <= 0).any():
        vols = vols.replace(0.0, np.nan)
        vols = vols.fillna(vols.median() if vols.median() > 0 else 0.15)
    
    diag = np.diag(vols.to_numpy(dtype=float))
    cov = diag @ corr.to_numpy(dtype=float) @ diag
    cov = cov + np.eye(len(corr.index)) * 1e-8
    return pd.DataFrame(cov, index=corr.index, columns=corr.columns)


def optimize_portfolio(
    expected_returns,
    covariance_matrix,
    max_weight=DEFAULT_MAX_WEIGHT,
    min_weight=DEFAULT_MIN_WEIGHT,
    risk_aversion=1.0,
    weak_signal_threshold=0.0005,
    allow_short=False,
    net_exposure=1.0,
    target_vol=None,
    use_equal_weight_fallback=False,
):
    """Optimize portfolio weights using mean-variance optimization."""
    tickers = list(expected_returns.index)
    mu = expected_returns.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    cov = covariance_matrix.reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy(dtype=float)
    n_assets = len(tickers)

    if n_assets == 0:
        return pd.Series(dtype=float)

    if np.allclose(cov, 0):
        cov = np.eye(n_assets) * 1e-6

    def objective(weights):
        expected_portfolio_return = float(np.dot(mu, weights))
        portfolio_variance = float(weights.T @ cov @ weights)
        penalty = 0.001 * np.sum(weights**2)
        return -(expected_portfolio_return - risk_aversion * portfolio_variance - penalty)

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - net_exposure}]
    
    if allow_short:
        lower_bound = min_weight
        bounds = [(lower_bound, max_weight) for _ in range(n_assets)]
    else:
        lower_bound = 0.0
        bounds = [(lower_bound, max_weight) for _ in range(n_assets)]

    if allow_short:
        centered = mu - np.mean(mu)
        if np.allclose(centered, 0.0) or np.sum(np.abs(centered)) == 0:
            initial_weights = np.repeat(net_exposure / n_assets if n_assets > 0 else 0.0, n_assets)
        else:
            initial_weights = centered / np.sum(np.abs(centered))
            initial_weights = np.clip(initial_weights, lower_bound, max_weight)
            if np.isclose(net_exposure, 0.0):
                initial_weights = initial_weights - initial_weights.mean()
    else:
        initial_weights = np.repeat(1.0 / n_assets, n_assets)

    prediction_strength = float(np.nanmean(np.abs(mu)))
    weak_signal = not np.isfinite(prediction_strength) or prediction_strength < weak_signal_threshold
    
    if use_equal_weight_fallback or weak_signal:
        weights = equal_weights(tickers)
        if allow_short and np.isclose(net_exposure, 0.0):
            weights = weights - weights.mean()
        return weights

    try:
        result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints, options={'ftol': 1e-9, 'disp': False})
        
        if (not result.success) or np.any(np.isnan(result.x)):
            weights = equal_weights(tickers)
            if allow_short and np.isclose(net_exposure, 0.0):
                weights = weights - weights.mean()
            return weights
        
        weights = result.x
        weights = np.clip(weights, lower_bound, max_weight)
        weight_sum = weights.sum()
        
        if weight_sum <= 0:
            if allow_short and np.isclose(net_exposure, 0.0):
                weights = np.zeros(n_assets)
            else:
                weights = np.repeat(1.0 / n_assets, n_assets)
        else:
            if not np.isclose(net_exposure, 0.0):
                weights = weights / weight_sum * net_exposure
        
        if target_vol is not None and np.isfinite(target_vol) and target_vol > 0:
            portfolio_vol = float(np.sqrt(weights.T @ cov @ weights))
            if portfolio_vol > 0:
                scale = target_vol / portfolio_vol
                scale = min(scale, 3.0)
                weights = weights * scale
                
    except Exception:
        weights = equal_weights(tickers)
        if allow_short and np.isclose(net_exposure, 0.0):
            weights = weights - weights.mean()
    
    return pd.Series(weights, index=tickers)


def run_simple_momentum_backtest(actual_returns, monthly_returns, sector_tickers=SECTOR_TICKERS, lookback=6, use_vol_scaling=True):
    """Simple momentum backtest for comparison."""
    portfolio_value = 1.0
    returns_list = []
    port_ret_history = []

    for date in actual_returns.index:
        history = monthly_returns.loc[monthly_returns.index < date].tail(lookback)
        if len(history) < lookback:
            weights = equal_weights(sector_tickers)
        else:
            momentum = history.mean().reindex(sector_tickers).fillna(0.0)
            ranks = momentum.rank(method="average", ascending=True)
            weights = ranks / ranks.sum()

        if use_vol_scaling and len(port_ret_history) > 60:
            forecast_vol = get_portfolio_volatility_forecast_no_lookahead(pd.Series(port_ret_history), current_position=len(port_ret_history))
            if forecast_vol > 0:
                scale = min(1.5, DEFAULT_TARGET_VOL / forecast_vol)
                weights = weights * scale

        actual = actual_returns.loc[date].reindex(sector_tickers).fillna(0.0)
        gross_ret = float((weights * actual).sum())
        portfolio_value *= (1 + gross_ret)
        returns_list.append(gross_ret)
        port_ret_history.append(gross_ret)

    returns_series = pd.Series(returns_list, index=actual_returns.index)
    equity_series = (1 + returns_series).cumprod()
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(12) if returns_series.std() > 0 else 0
    max_dd = ((equity_series / equity_series.cummax()) - 1).min()
    win_rate = (returns_series > 0).mean()
    total_return = equity_series.iloc[-1] - 1 if len(equity_series) > 0 else 0.0
    
    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_return": total_return,
        "returns_series": returns_series,
        "equity_series": equity_series,
    }


def run_backtest(predictions, actual_returns, monthly_returns, sector_tickers=SECTOR_TICKERS, use_vol_scaling=True):
    """Original backtest with volatility scaling."""
    portfolio_value = 1.0
    equity_curve = [1.0]
    returns_list = []
    trade_log_local = []
    prev_weights = pd.Series(0.0, index=sector_tickers)
    port_ret_history = []

    for date in predictions.index:
        pred = predictions.loc[date].reindex(sector_tickers).fillna(0.0)
        actual = actual_returns.loc[date].reindex(sector_tickers).fillna(0.0)

        covariance_matrix = estimate_covariance_matrix(monthly_returns, date, lookback_months=DEFAULT_LOOKBACK_MONTHS)
        weights = optimize_portfolio(pred, covariance_matrix, max_weight=DEFAULT_MAX_WEIGHT, risk_aversion=1.0)

        if use_vol_scaling and len(port_ret_history) > 60:
            forecast_vol = get_portfolio_volatility_forecast_no_lookahead(pd.Series(port_ret_history), current_position=len(port_ret_history))
            if forecast_vol > 0:
                scale = min(1.5, DEFAULT_TARGET_VOL / forecast_vol)
                weights = weights * scale

        weights = weights.clip(-0.33, 0.33)
        turnover = (weights - prev_weights).abs().sum() / 2
        tc = turnover * DEFAULT_TX_COST

        gross_ret = float((weights * actual).sum())
        net_ret = gross_ret - tc

        top_allocations = weights.sort_values(ascending=False).head(3)
        top_holdings = "|".join(top_allocations.index.tolist())
        top_weights = "|".join([f"{value:.3f}" for value in top_allocations.values])

        trade_log_local.append({
            "date": date,
            "holding_month": date + pd.DateOffset(months=1),
            "top_holdings": top_holdings,
            "top_weights": top_weights,
            "predicted_portfolio_return": float(np.dot(weights.values, pred.values)),
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
        })

        if date.month in [1, 7]:
            print(f"\n{date.strftime('%Y-%m')} | Top holdings: {top_holdings}")
            print(f"   Portfolio forecast: {trade_log_local[-1]['predicted_portfolio_return'] * 100:.3f}%")
            print(f"   Net ret: {net_ret * 100:.2f}%")

        portfolio_value *= (1 + net_ret)
        equity_curve.append(portfolio_value)
        returns_list.append(net_ret)
        port_ret_history.append(net_ret)
        prev_weights = weights.copy()

    returns_series = pd.Series(returns_list, index=predictions.index)
    equity_series = pd.Series(equity_curve[1:], index=predictions.index)
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(12) if returns_series.std() > 0 else 0
    max_dd = ((equity_series / equity_series.cummax()) - 1).min()
    win_rate = (returns_series > 0).mean()
    total_return = (portfolio_value - 1)

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_return": total_return,
        "returns_series": returns_series,
        "equity_series": equity_series,
        "trade_log_df": pd.DataFrame(trade_log_local).set_index("date"),
    }


def run_forecast_driven_backtest(
    predictions,
    actual_returns,
    monthly_returns,
    sector_vol_forecasts,
    sector_tickers=SECTOR_TICKERS,
    max_weight=DEFAULT_MAX_WEIGHT,
    min_weight=DEFAULT_MIN_WEIGHT,
    allow_short=ALLOW_SHORT,
    net_exposure=NET_EXPOSURE,
    target_vol=DEFAULT_TARGET_VOL,
    risk_aversion=OPTIMIZER_RISK_AVERSION,
):
    """Backtest with volatility forecasts integrated into portfolio optimization."""
    portfolio_value = 1.0
    equity_curve = [1.0]
    returns_list = []
    trade_log_local = []
    prev_weights = pd.Series(0.0, index=sector_tickers)

    for date in predictions.index:
        if date not in sector_vol_forecasts.index:
            continue
            
        pred = predictions.loc[date].reindex(sector_tickers).fillna(0.0)
        actual = actual_returns.loc[date].reindex(sector_tickers).fillna(0.0)
        forecast_vols = sector_vol_forecasts.loc[date].reindex(sector_tickers)
        
        forecast_vols = forecast_vols.fillna(forecast_vols.median() if forecast_vols.median() > 0 else 0.15)

        covariance_matrix = build_covariance_from_forecasts(monthly_returns, date, forecast_vols)
        
        weights = optimize_portfolio(
            pred,
            covariance_matrix,
            max_weight=max_weight,
            min_weight=min_weight,
            risk_aversion=risk_aversion,
            allow_short=allow_short,
            net_exposure=net_exposure,
            target_vol=target_vol,
        ).reindex(sector_tickers).fillna(0.0)

        turnover = (weights - prev_weights).abs().sum() / 2
        tc = turnover * DEFAULT_TX_COST
        gross_ret = float((weights * actual).sum())
        net_ret = gross_ret - tc

        top_allocations = weights.sort_values(ascending=False).head(3)
        top_holdings = "|".join(top_allocations.index.tolist())
        top_weights = "|".join([f"{value:.3f}" for value in top_allocations.values])
        
        cov_matrix = covariance_matrix.reindex(index=sector_tickers, columns=sector_tickers).fillna(0.0).to_numpy(dtype=float)
        forecast_portfolio_vol = float(np.sqrt(weights.to_numpy(dtype=float).T @ cov_matrix @ weights.to_numpy(dtype=float)))

        trade_log_local.append({
            "date": date,
            "holding_month": date + pd.DateOffset(months=1),
            "top_holdings": top_holdings,
            "top_weights": top_weights,
            "predicted_portfolio_return": float(np.dot(weights.values, pred.values)),
            "forecast_portfolio_vol": forecast_portfolio_vol,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
        })

        if date.month in [1, 7]:
            print(f"\n{date.strftime('%Y-%m')} | Top holdings: {top_holdings}")
            print(f"   Forecast return: {trade_log_local[-1]['predicted_portfolio_return'] * 100:.3f}%")
            print(f"   Forecast vol: {forecast_portfolio_vol * 100:.2f}%")
            print(f"   Net ret: {net_ret * 100:.2f}%")

        portfolio_value *= (1 + net_ret)
        equity_curve.append(portfolio_value)
        returns_list.append(net_ret)
        prev_weights = weights.copy()

    returns_series = pd.Series(returns_list, index=predictions.index[predictions.index.isin([t["date"] for t in trade_log_local])])
    equity_series = pd.Series(equity_curve[1:], index=returns_series.index)
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(12) if returns_series.std() > 0 else 0
    max_dd = ((equity_series / equity_series.cummax()) - 1).min()
    win_rate = (returns_series > 0).mean()
    total_return = portfolio_value - 1

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_return": total_return,
        "returns_series": returns_series,
        "equity_series": equity_series,
        "trade_log_df": pd.DataFrame(trade_log_local).set_index("date"),
    }