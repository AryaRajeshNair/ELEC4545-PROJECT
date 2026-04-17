from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from config import SECTOR_TICKERS
from data_prep import safe_get
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
# Explicit list so training and inference are guaranteed to use the same feature order.
SECTOR_FEATURES = [
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "mom_12m",
    "vol_3m",
    "vol_6m",
    "vol_12m",
    "rev_1m",
]

RF_PARAM_DIST = {
    'n_estimators': [200, 300, 500],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, None],
}


def prepare_training_data(
    features,
    returns,
    vix_norm,
    tnx_norm,
    current_date,
    sector_tickers=SECTOR_TICKERS,
    window_months=48,
):
    """Prepare training data for Random Forest."""
    # Validate inputs
    if not isinstance(features, dict):
        raise TypeError(f"features must be a dict, got {type(features)}")
    if SECTOR_FEATURES[0] not in features:
        raise KeyError(f"Missing required feature: {SECTOR_FEATURES[0]}. Available: {list(features.keys())}")
    
    current_date = pd.Timestamp(current_date).normalize()
    train_end = current_date - pd.DateOffset(months=1)
    train_start = train_end - pd.DateOffset(months=window_months)

    x_list, y_list = [], []
    
    # Use the first available feature's index (all features should be aligned)
    feature_df = features[SECTOR_FEATURES[0]]
    train_dates = feature_df.index[
        (feature_df.index >= train_start) & (feature_df.index <= train_end)
    ]

    next_return = returns.shift(-1)

    for date in train_dates:
        date = pd.Timestamp(date).normalize()

        for sector in sector_tickers:
            x = [
                safe_get(features[feat], date, sector) for feat in SECTOR_FEATURES
            ]

            if np.any(np.isnan(x)):
                continue

            if date in next_return.index:
                y = safe_get(next_return, date, sector)
                if np.isnan(y):
                    continue
                x_list.append(x)
                y_list.append(y)

    if len(x_list) == 0:
        return np.array([]), np.array([])
    
    return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


def train_with_hyperparameter_tuning(x_train, y_train, random_state=42, verbose=True):
    """
    Train Random Forest with RandomizedSearchCV and TimeSeriesSplit.
    
    Based on research papers:
    - "Optimizing Random Forest Hyperparameters for Enhanced Stock Price Prediction" (2025)
    - "Feature Importance Guided Random Forest Learning..." (arXiv 2511.00133)
    """
    # Base model
    base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    # TimeSeriesSplit prevents look-ahead bias (crucial for financial data!)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # RandomizedSearchCV with time-series cross-validation
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=RF_PARAM_DIST,
        n_iter=20,                    # Try 20 random combinations
        cv=tscv,                      # Time-series aware CV
        scoring='neg_mean_squared_error',  # Optimize for MSE
        n_jobs=-1,
        random_state=random_state,
        verbose=0 if not verbose else 1,
    )
    
    search.fit(x_train, y_train)
    
    if verbose:
        print(f"    Best params: {search.best_params_}")
        print(f"    Best CV score: {-search.best_score_:.6f} (MSE)")
    
    return search.best_estimator_, search.best_params_


def run_rolling_predictions(
    features,
    monthly_returns,
    vix_norm,
    tnx_norm,
    sector_tickers=SECTOR_TICKERS,
    test_start="2020-01-01",
    train_window_months=48,
    min_training_samples=100,
    random_state=42,
    use_hyperparameter_tuning=True,  # NEW: Enable/disable tuning
):
    """Run rolling window predictions using Random Forest with hyperparameter tuning."""
    print("\n[STEP 4] Training Random Forest...")
    if use_hyperparameter_tuning:
        print("  Using RandomizedSearchCV with TimeSeriesSplit (research-backed tuning)")
    else:
        print("  Using fixed hyperparameters")

    test_dates = [
        dt
        for dt in monthly_returns.index
        if dt >= pd.Timestamp(test_start) and dt < monthly_returns.index.max()
    ]

    predictions_list = []
    tuning_results = []
    
    for i, current_date in enumerate(test_dates):
        if i % 12 == 0 or i == len(test_dates) - 1:
            print(f" Processing {current_date.strftime('%Y-%m')} ({i+1}/{len(test_dates)})")

        x_train, y_train = prepare_training_data(
            features,
            monthly_returns,
            vix_norm,
            tnx_norm,
            current_date,
            sector_tickers,
            train_window_months,
        )

        if len(x_train) < min_training_samples:
            predictions_list.append({
                "date": current_date, 
                **{sector: np.nan for sector in sector_tickers}
            })
            continue

        if use_hyperparameter_tuning and len(x_train) >= 200:
            model, best_params = train_with_hyperparameter_tuning(
                x_train, y_train, random_state, verbose=False
            )
            tuning_results.append({
                'date': current_date,
                'params': best_params,
                'n_samples': len(x_train)
            })
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)

        pred_dict = {"date": current_date}
        for sector in sector_tickers:
            x = [
                safe_get(features[feat], current_date, sector) for feat in SECTOR_FEATURES
            ]

            if np.any(np.isnan(x)):
                pred_dict[sector] = np.nan
            else:
                pred_dict[sector] = float(model.predict([x])[0])

        predictions_list.append(pred_dict)

    predictions_df = pd.DataFrame(predictions_list).set_index("date")
    
    nan_count = predictions_df.isna().sum().sum()
    total_predictions = predictions_df.size
    print(f"\n✓ Generated predictions for {len(predictions_df)} months")
    print(f"  ({nan_count}/{total_predictions} predictions are NaN due to missing data)")
    
    if tuning_results:
        print(f"\n✓ Hyperparameter tuning completed for {len(tuning_results)} rolling windows")
        param_counts = {}
        for r in tuning_results:
            for k, v in r['params'].items():
                key = f"{k}={v}"
                param_counts[key] = param_counts.get(key, 0) + 1
        
        print("\n  Most frequently selected parameters:")
        for param, count in sorted(param_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    {param}: {count}/{len(tuning_results)} windows")
    
    return predictions_df


def evaluate_predictions(predictions_df, actual_returns, sector_tickers=SECTOR_TICKERS):
    """Evaluate prediction accuracy."""
    print("\n" + "=" * 60)
    print("PREDICTION EVALUATION")
    print("=" * 60)

    correct = total = 0
    all_preds, all_actuals = [], []

    for date in predictions_df.index:
        for sector in sector_tickers:
            pred = predictions_df.loc[date, sector]
            actual = actual_returns.loc[date, sector]
            if pd.notna(pred) and pd.notna(actual):
                if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
                    correct += 1
                total += 1
                all_preds.append(pred)
                all_actuals.append(actual)

    directional_accuracy = correct / total if total > 0 else np.nan
    correlation = (
        np.corrcoef(all_preds, all_actuals)[0, 1] if len(all_preds) > 1 else np.nan
    )

    print(f"Valid predictions   : {total}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")
    print(f"Correlation         : {correlation:.4f}")

    return {
        "directional_accuracy": directional_accuracy,
        "correlation": correlation,
        "correct": correct,
        "total": total,
        "all_preds": all_preds,
        "all_actuals": all_actuals,
    }


def test_prediction_value(predictions_df, actual_returns):
    """Test whether predictions can sort sectors into better/worse future returns."""
    spread_returns = []
    
    for date in predictions_df.index:
        preds = predictions_df.loc[date]
        actuals = actual_returns.loc[date]
        valid = pd.DataFrame({"pred": preds, "actual": actuals}).dropna()
        if len(valid) < 4:
            continue

        sorted_valid = valid.sort_values("pred")
        top_n = max(1, int(len(sorted_valid) * 0.2))
        top_return = sorted_valid["actual"].tail(top_n).mean()
        bottom_return = sorted_valid["actual"].head(top_n).mean()
        spread_returns.append(top_return - bottom_return)

    spread_series = pd.Series(spread_returns, dtype=float)
    
    if len(spread_series) < 3:
        print("\nPrediction Sorting Test: insufficient observations.")
        return np.nan, np.nan

    t_stat, p_value = stats.ttest_1samp(spread_series, 0.0, nan_policy="omit")

    print("\n" + "=" * 60)
    print("PREDICTION SORTING TEST")
    print("=" * 60)
    print(f"Mean top-bottom spread: {spread_series.mean() * 100:.3f}%")
    print(f"t-statistic           : {t_stat:.3f}")
    print(f"p-value               : {p_value:.4f}")

    if p_value < 0.05 and spread_series.mean() > 0:
        print("✓ Predictions have statistically significant sorting power")
    else:
        print("✗ Predictions cannot sort sectors effectively")

    return float(spread_series.mean()), float(p_value)


def get_feature_importance(model, feature_names=None):
    """Get feature importance from trained Random Forest."""
    if feature_names is None:
        feature_names = [
            'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m',
            'vol_3m', 'vol_6m', 'vol_12m', 'rev_1m'
        ]
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 60)
    
    for _, row in importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:12s}: {row['importance']:.4f}  {bar}")
    
    return importance


def calculate_average_feature_importance(
    predictions_df, 
    features, 
    monthly_returns, 
    vix_norm, 
    tnx_norm, 
    sector_tickers=SECTOR_TICKERS,
    n_samples=12
):
    """Calculate average feature importance across multiple rolling windows."""
    from sklearn.ensemble import RandomForestRegressor
    
    all_importance = []
    test_dates = predictions_df.index
    
    sample_dates = test_dates[:min(n_samples, len(test_dates))]
    
    print(f"\nCalculating feature importance across {len(sample_dates)} rolling windows...")
    
    for i, current_date in enumerate(sample_dates):
        print(f"  Window {i+1}/{len(sample_dates)}: {current_date.strftime('%Y-%m')}")
        
        x_train, y_train = prepare_training_data(
            features, monthly_returns, vix_norm, tnx_norm, 
            current_date, sector_tickers
        )
        
        if len(x_train) < 100:
            print(f"    Skipped (only {len(x_train)} training samples)")
            continue
        
        # Use improved hyperparameters
        model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5, 
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42, 
            n_jobs=-1
        )
        model.fit(x_train, y_train)
        all_importance.append(model.feature_importances_)
    
    if not all_importance:
        print("Not enough data for feature importance calculation")
        return None
    
    avg_importance = np.mean(all_importance, axis=0)
    std_importance = np.std(all_importance, axis=0)
    
    feature_names = [
        'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 
        'vol_3m', 'vol_6m', 'vol_12m', 'rev_1m'
    ]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance,
        'std_dev': std_importance
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 60)
    print("AVERAGE FEATURE IMPORTANCE (across rolling windows)")
    print("=" * 60)
    for _, row in importance_df.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:12s}: {row['importance']:.4f} ± {row['std_dev']:.4f}  {bar}")
    
    # Identify top features
    print("\n" + "-" * 60)
    print("Top 3 most important features:")
    for i, row in importance_df.head(3).iterrows():
        print(f"  {i+1}. {row['feature']} ({row['importance']:.4f})")
    
    return importance_df


def calculate_feature_importance_over_time(
    features, 
    monthly_returns, 
    vix_norm, 
    tnx_norm, 
    test_dates, 
    sector_tickers=SECTOR_TICKERS
):
    """Calculate how feature importance evolves over time (for concept drift analysis)."""
    from sklearn.ensemble import RandomForestRegressor
    
    feature_names = [
        'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 
        'vol_3m', 'vol_6m', 'vol_12m', 'rev_1m'
    ]
    
    importance_over_time = []
    
    print("\nCalculating feature importance evolution over time...")
    
    for i, current_date in enumerate(test_dates):
        if i % 12 == 0:  # Print every 12 months
            print(f"  Processing {current_date.strftime('%Y-%m')}")
        
        x_train, y_train = prepare_training_data(
            features, monthly_returns, vix_norm, tnx_norm, 
            current_date, sector_tickers
        )
        
        if len(x_train) < 100:
            continue
        
        model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5, 
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42, 
            n_jobs=-1
        )
        model.fit(x_train, y_train)
        
        row = {'date': current_date}
        for j, name in enumerate(feature_names):
            row[name] = model.feature_importances_[j]
        importance_over_time.append(row)
    
    if not importance_over_time:
        return None
    
    df = pd.DataFrame(importance_over_time).set_index('date')
    
    
    print("\nFeature importance stability check:")
    for col in df.columns:
        cv = df[col].std() / df[col].mean() if df[col].mean() > 0 else np.nan
        print(f"  {col:12s}: CV = {cv:.3f} ({'stable' if cv < 0.5 else 'unstable'})")
    
    return df