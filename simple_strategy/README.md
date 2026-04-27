# Time Series Analysis and Financial Applications

A quantitative trading system that combines **GARCH volatility forecasting** and **machine learning** (Random Forest) to predict sector returns and optimize portfolio allocation.

## Overview

This project implements a comprehensive quantitative strategy that:

1. **Downloads market data** for 11 US sector ETFs (XLC, XLY, XLP, XLE, XLF, XLI, XLB, XLK, XLV, XLU, XLRE)
2. **Computes technical features** including momentum (1m, 3m, 6m, 12m), volatility, and reversals
3. **Forecasts volatility** using GARCH(1,1) models for risk management
4. **Trains a Random Forest** with hyperparameter tuning to predict next-month sector returns
5. **Backtests two strategies**:
   - **Simple Momentum**: Equal-weight positions on positive momentum signals
   - **ML-Driven**: Portfolio optimized using ML predictions and GARCH volatility forecasts
6. **Generates performance reports** with metrics, trade logs, and feature importance analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd simple_strategy

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

This will execute all steps and save results to the `results/` directory.

## Project Structure

```
simple_strategy/
├── config.py                 # Configuration (tickers, dates, parameters)
├── data_prep.py             # Download market data & compute features
├── garch.py                 # GARCH volatility models & diagnostics
├── ml_model.py              # Random Forest with hyperparameter tuning
├── portfolio.py             # Portfolio optimization & backtesting
├── main.py                  # Main pipeline orchestrator
├── requirements.txt         # Python dependencies
├── figures/                 # Generated plots & visualizations
└── results/                 # CSV outputs (predictions, metrics, trades)
```

## Module Descriptions

### `config.py`
Central configuration file containing:
- Sector tickers and date ranges
- Hyperparameters (max/min weights, risk aversion, transaction costs)
- Directory paths for outputs

### `data_prep.py`
- Downloads OHLCV data from Yahoo Finance
- Computes cross-sectional normalized features:
  - **Momentum**: 1m, 3m, 6m, 12m returns
  - **Volatility**: 3m, 6m, 12m rolling std
  - **Reversals**: 1m reversal signal
- Ensures aligned indices across all sectors and dates

### `garch.py`
- Fits GARCH(1,1) models to sector returns
- Generates next-month volatility forecasts
- Performs diagnostics: Ljung-Box tests, ARCH tests, ADF/KPSS stationarity
- Aggregates forecasts for portfolio-level risk estimates

### `ml_model.py`
- **Hyperparameter tuning**: RandomizedSearchCV with TimeSeriesSplit (prevents look-ahead bias)
- **Training**: Rolling window approach with 48-month training windows
- **Predictions**: Next-month return forecasts for each sector
- **Evaluation**: Directional accuracy, correlation, sorting power tests
- **Feature importance**: Tracks which features drive predictions over time

### `portfolio.py`
- **Momentum backtest**: Equal-weight long positions on positive momentum
- **ML-driven backtest**: Mean-variance optimization using:
  - ML return predictions as expected returns
  - GARCH forecasts as volatility estimates
  - Sector correlations from recent data
- **Risk management**: Position sizing, transaction costs, constraints
- **Metrics**: Sharpe ratio, max drawdown, win rate, trade statistics

### `main.py`
Orchestrates the full pipeline:
1. Download market data
2. Generate features
3. Fit GARCH models
4. Train Random Forest with rolling windows
5. Run both backtests
6. Generate summary reports and visualizations

## Output Files

All results saved to `results/`:

- **`backtest_summary_table.csv`** — Performance metrics (Sharpe, returns, drawdown, etc.)
- **`backtest_monthly_results.csv`** — Month-by-month PnL and risk metrics
- **`trade_log.csv`** — Detailed record of all trades
- **`rf_predictions.csv`** — ML model predictions for each sector/month
- **`sector_vol_forecasts.csv`** — GARCH volatility forecasts
- **`feature_importance.csv`** — Feature contribution to predictions
- **`garch_diagnostics_summary.csv`** — GARCH model diagnostics

Plots saved to `figures/`:
- Equity curves and drawdowns
- Feature importance visualizations
- Prediction accuracy charts
- GARCH diagnostics plots

## Key Features

### Research-Backed Design
- **TimeSeriesSplit** cross-validation prevents look-ahead bias
- **GARCH(1,1)** models capture volatility clustering in returns
- **Rank normalization** reduces cross-sectional biases
- **Mean-variance optimization** with realistic constraints

### Hyperparameter Tuning
The Random Forest hyperparameters are tuned on each rolling window:
- `n_estimators`: [200, 300, 500]
- `max_depth`: [None, 10, 15, 20]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2', 0.5, None]

### Evaluation Metrics
- **Directional Accuracy**: % of correct next-month return direction predictions
- **Correlation**: Predicted vs. actual returns
- **Sorting Power**: Can predictions rank sectors from best to worst future performers?
- **Backtest Metrics**: Sharpe ratio, CAGR, max drawdown, Calmar ratio, win rate

## Usage Example

```python
from config import SECTOR_TICKERS, START_DATE, TEST_START
from data_prep import download_market_data, create_features
from garch import forecast_sector_volatility_garch
from ml_model import run_rolling_predictions, evaluate_predictions
from portfolio import run_forecast_driven_backtest

# Download data
prices = download_market_data(SECTOR_TICKERS, START_DATE)

# Generate features
features, returns, vix_norm, tnx_norm = create_features(prices)

# Forecast volatility
sector_vol_forecasts = forecast_sector_volatility_garch(returns)

# Train ML model
predictions = run_rolling_predictions(
    features, returns, vix_norm, tnx_norm, 
    test_start=TEST_START
)

# Evaluate
eval_results = evaluate_predictions(predictions, returns)

# Backtest
backtest_results = run_forecast_driven_backtest(
    predictions, returns, sector_vol_forecasts
)
```

## Configuration

Edit `config.py` to customize:

```python
# Asset universe
SECTOR_TICKERS = ["XLC", "XLY", ...]  # Add/remove sectors

# Date ranges
START_DATE = "2015-01-01"
TEST_START = "2020-01-01"
TRAIN_WINDOW_MONTHS = 48

# Portfolio constraints
DEFAULT_MAX_WEIGHT = 0.20
DEFAULT_MIN_WEIGHT = -0.20  # Set to 0 to disable shorts
ALLOW_SHORT = False
NET_EXPOSURE = 1.0  # Full investment

# Risk parameters
DEFAULT_TARGET_VOL = 0.12
OPTIMIZER_RISK_AVERSION = 2.0
DEFAULT_TX_COST = 0.001
```

## Dependencies

- **numpy, pandas, scipy** — Data processing and numerical computing
- **scikit-learn** — Random Forest and hyperparameter tuning
- **statsmodels, arch** — Time series and GARCH models
- **yfinance** — Market data acquisition
- **matplotlib** — Visualization

See `requirements.txt` for versions.

## Performance Notes

- Full pipeline typically takes **10-30 minutes** depending on data volume and CPU cores
- Hyperparameter tuning is parallelized via `n_jobs=-1`
- Disable tuning by setting `use_hyperparameter_tuning=False` in `run_rolling_predictions()` for faster runs

## Research References

The methodology draws from:
- "Optimizing Random Forest Hyperparameters for Enhanced Stock Price Prediction" (2025)
- "Feature Importance Guided Random Forest Learning..." (arXiv 2511.00133)
- GARCH models for volatility forecasting in asset management
- Mean-variance portfolio optimization (Markowitz framework)

## Disclaimer

This is an educational/research project. Past performance is not indicative of future results. Use at your own risk. Always validate strategies with adequate backtesting and live-trading validation.

## License

[Specify your license here]

## Contact

[Add contact info if desired]
