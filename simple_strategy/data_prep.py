from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from config import SECTOR_TICKERS, START_DATE


def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalization (values between -1 and 1)."""
    if df.isna().all().all():
        return pd.DataFrame(0.0, index=df.index, columns=df.columns)
    
    ranks = df.rank(axis=1, method="average", pct=True)
    ranks = ranks.fillna(0.5)
    return (ranks * 2 - 1).fillna(0)


def safe_get(df, date, col=None):
    """Safely extract a value from a DataFrame or Series."""
    try:
        date = pd.Timestamp(date).normalize()
        
        if col is not None:
            if date in df.index and col in df.columns:
                value = df.loc[date, col]
            else:
                return np.nan
        else:
            if isinstance(df, pd.DataFrame):
                if len(df.columns) == 1:
                    if date in df.index:
                        value = df.loc[date].iloc[0]
                    else:
                        return np.nan
                else:
                    if date in df.index:
                        row = df.loc[date]
                        if isinstance(row, pd.Series) and len(row) > 0:
                            value = row.iloc[0]
                        else:
                            value = row
                    else:
                        return np.nan
            else:
                if date in df.index:
                    value = df.loc[date]
                else:
                    return np.nan
        
        return float(value) if pd.notna(value) else np.nan
    except Exception:
        return np.nan


def compute_rsi(prices, period=14):
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices, fast=12, slow=26, signal=9):
    """Compute MACD and MACD Signal line."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal


def compute_bollinger_bands(prices, period=20, std_dev=2):
    """Compute Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower


def compute_rate_of_change(prices, periods=[1, 3, 6]):
    """Compute Rate of Change for multiple periods."""
    roc_dict = {}
    for period in periods:
        roc_dict[f"roc_{period}"] = (prices - prices.shift(period)) / prices.shift(period) * 100
    return roc_dict


def download_market_data(sector_tickers=SECTOR_TICKERS, start_date=START_DATE, end_date=None):
    """Download OHLC data for sector ETFs, VIX, and Treasury yields."""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"Downloading data from {start_date} to {end_date}...")
    
    # Download sector ETFs with OHLC data (auto_adjust=False to get High/Low)
    data = yf.download(
        sector_tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    daily_close = pd.DataFrame()
    daily_high = pd.DataFrame()
    daily_low = pd.DataFrame()
    daily_volume = pd.DataFrame()
    
    for ticker in sector_tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                daily_close[ticker] = data[ticker]["Close"]
                daily_high[ticker] = data[ticker]["High"]
                daily_low[ticker] = data[ticker]["Low"]
                daily_volume[ticker] = data[ticker]["Volume"]
            else:
                daily_close[ticker] = data["Close"][ticker] if ticker in data["Close"].columns else pd.NA
                daily_high[ticker] = data["High"][ticker] if ticker in data["High"].columns else pd.NA
                daily_low[ticker] = data["Low"][ticker] if ticker in data["Low"].columns else pd.NA
                daily_volume[ticker] = data["Volume"][ticker] if ticker in data["Volume"].columns else pd.NA
        except Exception as e:
            print(f"Warning: Could not download {ticker}: {e}")
            daily_close[ticker] = pd.NA
            daily_high[ticker] = pd.NA
            daily_low[ticker] = pd.NA
            daily_volume[ticker] = pd.NA

    # Download VIX and TNX
    print("Downloading VIX and TNX...")
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)["Close"]
    tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)["Close"]

    # Resample to month-end
    monthly_close = daily_close.resample("ME").last()
    monthly_high = daily_high.resample("ME").max()
    monthly_low = daily_low.resample("ME").min()
    monthly_volume = daily_volume.resample("ME").sum()
    monthly_returns = monthly_close.pct_change().dropna(how="all")
    
    # Align VIX and TNX to monthly returns index
    monthly_vix = vix.resample("ME").last()
    monthly_vix = monthly_vix.reindex(monthly_returns.index).ffill()
    
    monthly_tnx = tnx.resample("ME").last()
    monthly_tnx = monthly_tnx.reindex(monthly_returns.index).ffill()

    # Remove any rows with all NaN returns
    monthly_returns = monthly_returns.dropna(how="all")

    print(f"Data downloaded: {len(monthly_returns)} months of returns")
    print(f"Date range: {monthly_returns.index[0]} to {monthly_returns.index[-1]}")

    return {
        "daily_close": daily_close,
        "daily_high": daily_high,
        "daily_low": daily_low,
        "daily_volume": daily_volume,
        "monthly_close": monthly_close,
        "monthly_high": monthly_high,
        "monthly_low": monthly_low,
        "monthly_volume": monthly_volume,
        "monthly_returns": monthly_returns,
        "monthly_vix": monthly_vix,
        "monthly_tnx": monthly_tnx,
    }


def create_features(
    monthly_close,
    monthly_returns,
    monthly_vix,
    monthly_tnx,
    min_periods=12,
):
    """
    Create features including technical indicators.
    
    Args:
        monthly_close: DataFrame of monthly closing prices
        monthly_returns: DataFrame of monthly returns
        monthly_vix: Series of VIX values
        monthly_tnx: Series of TNX values
        min_periods: Minimum periods for expanding window
    
    Returns:
        features: Dictionary of feature DataFrames
        vix_norm: Normalized VIX Series
        tnx_norm: Normalized TNX Series
    """
    features = {}
    
    # ========================================================================
    # MOMENTUM FEATURES
    # ========================================================================
    features["mom_1m"] = rank_normalize(monthly_returns)
    features["mom_3m"] = rank_normalize(monthly_returns.rolling(3, min_periods=3).mean())
    features["mom_6m"] = rank_normalize(monthly_returns.rolling(6, min_periods=6).mean())
    features["mom_12m"] = rank_normalize(monthly_returns.rolling(12, min_periods=12).mean())
    
    # ========================================================================
    # VOLATILITY FEATURES
    # ========================================================================
    features["vol_3m"] = rank_normalize(monthly_returns.rolling(3, min_periods=3).std())
    features["vol_6m"] = rank_normalize(monthly_returns.rolling(6, min_periods=6).std())
    features["vol_12m"] = rank_normalize(monthly_returns.rolling(12, min_periods=12).std())
    
    # ========================================================================
    # REVERSAL FEATURE
    # ========================================================================
    features["rev_1m"] = rank_normalize(-monthly_returns)
    
    # ========================================================================
    # TECHNICAL INDICATORS (Phase 1)
    # ========================================================================
    
    # RSI (14-period)
    rsi_values = pd.DataFrame(index=monthly_close.index, columns=monthly_close.columns)
    for sector in monthly_close.columns:
        rsi_values[sector] = compute_rsi(monthly_close[sector], period=14)
    features["rsi_14"] = rank_normalize(rsi_values)
    
    # MACD and MACD Signal
    macd_line_df = pd.DataFrame(index=monthly_close.index, columns=monthly_close.columns)
    macd_signal_df = pd.DataFrame(index=monthly_close.index, columns=monthly_close.columns)
    for sector in monthly_close.columns:
        macd_line, macd_signal = compute_macd(monthly_close[sector])
        macd_line_df[sector] = macd_line
        macd_signal_df[sector] = macd_signal
    features["macd"] = rank_normalize(macd_line_df)
    features["macd_signal"] = rank_normalize(macd_signal_df)
    
    # Bollinger Bands (Upper and Lower)
    bollinger_upper_df = pd.DataFrame(index=monthly_close.index, columns=monthly_close.columns)
    bollinger_lower_df = pd.DataFrame(index=monthly_close.index, columns=monthly_close.columns)
    for sector in monthly_close.columns:
        upper, lower = compute_bollinger_bands(monthly_close[sector])
        bollinger_upper_df[sector] = upper
        bollinger_lower_df[sector] = lower
    features["bollinger_upper"] = rank_normalize(bollinger_upper_df)
    features["bollinger_lower"] = rank_normalize(bollinger_lower_df)
    
    # Rate of Change (1, 3, 6 months)
    for sector in monthly_close.columns:
        roc_dict = compute_rate_of_change(monthly_close[sector], periods=[1, 3, 6])
        for roc_name, roc_values in roc_dict.items():
            col_name = f"price_{roc_name}"
            if col_name not in features:
                features[col_name] = pd.DataFrame(index=monthly_close.index, columns=monthly_close.columns)
            features[col_name].loc[:, sector] = roc_values
    
    # Rank normalize all ROC features
    for roc_name in ["price_roc_1", "price_roc_3", "price_roc_6"]:
        if roc_name in features:
            features[roc_name] = rank_normalize(features[roc_name])
    
    # ========================================================================
    # MACRO FEATURES (VIX and TNX normalization)
    # ========================================================================
    vix_mean_exp = monthly_vix.expanding(min_periods=min_periods).mean()
    vix_std_exp = monthly_vix.expanding(min_periods=min_periods).std()
    vix_std_exp = vix_std_exp.replace(0, np.nan)
    
    tnx_mean_exp = monthly_tnx.expanding(min_periods=min_periods).mean()
    tnx_std_exp = monthly_tnx.expanding(min_periods=min_periods).std()
    tnx_std_exp = tnx_std_exp.replace(0, np.nan)

    vix_norm = (monthly_vix - vix_mean_exp) / vix_std_exp
    tnx_norm = (monthly_tnx - tnx_mean_exp) / tnx_std_exp
    
    vix_norm = vix_norm.fillna(0).replace([np.inf, -np.inf], 0)
    tnx_norm = tnx_norm.fillna(0).replace([np.inf, -np.inf], 0)

    return features, vix_norm, tnx_norm


def prepare_features_for_ml(features, vix_norm, tnx_norm):
    """Prepare features for machine learning by aligning all data."""
    common_index = features["mom_1m"].index
    
    aligned_features = {}
    for key, df in features.items():
        aligned_features[key] = df.reindex(common_index)
    
    aligned_vix = vix_norm.reindex(common_index)
    aligned_tnx = tnx_norm.reindex(common_index)
    
    return aligned_features, aligned_vix, aligned_tnx


def validate_data(data_dict):
    """Validate that downloaded data is complete and has no major issues."""
    monthly_returns = data_dict["monthly_returns"]
    
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    
    missing_pct = monthly_returns.isna().mean().mean() * 100
    print(f"Missing returns: {missing_pct:.2f}%")
    
    if missing_pct > 10:
        print("WARNING: High percentage of missing returns!")
    
    for ticker in monthly_returns.columns:
        if monthly_returns[ticker].std() == 0:
            print(f"WARNING: {ticker} has zero variance (constant prices?)")
    
    print(f"Date range: {monthly_returns.index[0]} to {monthly_returns.index[-1]}")
    print(f"Total months: {len(monthly_returns)}")
    
    vix = data_dict["monthly_vix"]
    tnx = data_dict["monthly_tnx"]
    
    print(f"VIX available: {vix.notna().sum()} months")
    print(f"TNX available: {tnx.notna().sum()} months")
    
    return missing_pct < 20