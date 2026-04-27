"""
Figure generation module for quantitative trading strategy backtest results.
Generates 7 publication-quality figures and saves to figures/ directory.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Global styling
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# Color palette
BLUE = "#1F4E79"
MID_BLUE = "#2E75B6"
LIGHT_BLUE = "#D6E4F0"
ORANGE = "#C55A11"
GRAY = "#595959"
RED = "#C00000"
GREEN = "#375623"


def fig1_equity_curves(forecast_backtest, baseline_backtest, figures_dir):
    """Figure 1: Equity curves with drawdown shading."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract equity curves, normalized to 1.0 at start
    strategy_equity = forecast_backtest["equity_curve"].copy()
    baseline_equity = baseline_backtest["equity_curve"].copy()
    
    strategy_equity = strategy_equity / strategy_equity.iloc[0]
    baseline_equity = baseline_equity / baseline_equity.iloc[0]
    
    # Get dates and values
    dates = strategy_equity.index
    strategy_vals = strategy_equity.values
    baseline_vals = baseline_equity.values
    
    # Plot equity curves
    ax.plot(dates, strategy_vals, color=MID_BLUE, linewidth=2.0, 
            label="Strategy (RF + GARCH)")
    ax.plot(dates, baseline_vals, color=ORANGE, linewidth=1.5, 
            linestyle="--", label="Baseline (Momentum)")
    
    # Add horizontal line at y=1.0
    ax.axhline(y=1.0, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.7)
    
    # Compute and shade drawdowns for strategy
    cummax = np.maximum.accumulate(strategy_vals)
    drawdown = strategy_vals / cummax - 1
    drawdown_periods = drawdown < -0.03
    
    # Find contiguous drawdown periods
    drawdown_indices = np.where(drawdown_periods)[0]
    if len(drawdown_indices) > 0:
        # Group into contiguous blocks
        breaks = np.where(np.diff(drawdown_indices) > 1)[0]
        blocks = np.split(drawdown_indices, breaks + 1)
        
        for block in blocks:
            start_date = dates[block[0]]
            end_date = dates[block[-1]]
            ax.axvspan(start_date, end_date, color="#FADADD", alpha=0.3)
    
    # Formatting
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.set_title("Strategy vs Baseline — Cumulative Equity (Jan 2020 – Mar 2026)", 
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # Set y-axis to start slightly below minimum
    y_min = min(strategy_vals.min(), baseline_vals.min())
    y_buffer = (strategy_vals.max() - y_min) * 0.05
    ax.set_ylim(y_min - y_buffer, strategy_vals.max() * 1.05)
    
    # Final values annotation
    final_strategy = strategy_vals[-1]
    final_baseline = baseline_vals[-1]
    annotation_text = f"Strategy: {final_strategy:.2f}x  |  Baseline: {final_baseline:.2f}x"
    ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Legend placement
    if final_strategy < final_baseline:
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    else:
        ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    fig.savefig(figures_dir / "fig1_equity_curves.png", 
                bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print("  ✓ fig1_equity_curves.png")


def fig2_prediction_quality(predictions, actual_returns, figures_dir):
    """Figure 2: Prediction quality scatter + rolling accuracy."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Scatter plot of predictions vs actuals
    # Flatten all predictions and actuals
    pred_flat = []
    actual_flat = []
    for date in predictions.index:
        for sector in predictions.columns:
            pred = predictions.loc[date, sector]
            actual = actual_returns.loc[date, sector]
            if pd.notna(pred) and pd.notna(actual):
                pred_flat.append(pred)
                actual_flat.append(actual)
    
    pred_flat = np.array(pred_flat)
    actual_flat = np.array(actual_flat)
    
    # Scatter plot
    ax_left.scatter(pred_flat, actual_flat, alpha=0.25, s=15, color=MID_BLUE)
    
    # Fit regression line
    if len(pred_flat) > 1:
        z = np.polyfit(pred_flat, actual_flat, 1)
        p = np.poly1d(z)
        x_line = np.linspace(pred_flat.min(), pred_flat.max(), 100)
        ax_left.plot(x_line, p(x_line), color=ORANGE, linewidth=1.5)
    
    # Add zero lines
    ax_left.axhline(y=0, color=GRAY, linestyle="--", linewidth=0.7, alpha=0.7)
    ax_left.axvline(x=0, color=GRAY, linestyle="--", linewidth=0.7, alpha=0.7)
    
    # Compute metrics
    correlation = np.corrcoef(pred_flat, actual_flat)[0, 1]
    directional_accuracy = np.mean(
        (pred_flat > 0) & (actual_flat > 0) | (pred_flat < 0) & (actual_flat < 0)
    )
    
    # Annotation
    ann_text = f"Correlation: {correlation:.4f}\nDir. Accuracy: {directional_accuracy:.2%}\nN = {len(pred_flat)}"
    ax_left.text(0.05, 0.95, ann_text, transform=ax_left.transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    
    ax_left.set_xlabel("Predicted Return", fontsize=11)
    ax_left.set_ylabel("Realized Return", fontsize=11)
    ax_left.set_title("RF Prediction Quality (All Sectors, All Months)", 
                      fontsize=12, fontweight="bold")
    ax_left.grid(True, alpha=0.2)
    
    # Panel B: Rolling 12-month directional accuracy
    rolling_accuracy = []
    rolling_dates = []
    
    for i, current_date in enumerate(predictions.index):
        if i < 11:  # Need at least 12 months of history
            continue
        
        # Get previous 12 months
        window_start = predictions.index[i - 11]
        window_end = current_date
        
        pred_window = predictions.loc[window_start:window_end].values.flatten()
        actual_window = actual_returns.loc[window_start:window_end].values.flatten()
        
        # Remove NaNs
        valid_mask = ~(np.isnan(pred_window) | np.isnan(actual_window))
        pred_window = pred_window[valid_mask]
        actual_window = actual_window[valid_mask]
        
        if len(pred_window) > 0:
            acc = np.mean(
                (pred_window > 0) & (actual_window > 0) | 
                (pred_window < 0) & (actual_window < 0)
            )
            rolling_accuracy.append(acc)
            rolling_dates.append(current_date)
    
    rolling_accuracy = np.array(rolling_accuracy)
    rolling_dates = np.array(rolling_dates)
    
    # Plot line
    ax_right.plot(rolling_dates, rolling_accuracy, color=MID_BLUE, linewidth=1.8)
    
    # Add reference lines
    overall_avg = directional_accuracy
    ax_right.axhline(y=0.5, color=GRAY, linestyle="--", linewidth=1.0, 
                     label="Random (50%)", alpha=0.8)
    ax_right.axhline(y=overall_avg, color=ORANGE, linestyle=":", linewidth=1.0,
                     label=f"Overall avg ({overall_avg:.2%})", alpha=0.8)
    
    # Shade regions
    for i in range(len(rolling_dates) - 1):
        if rolling_accuracy[i] >= 0.5:
            ax_right.fill_between([rolling_dates[i], rolling_dates[i+1]], 
                                  0.5, rolling_accuracy[i], color=LIGHT_BLUE, alpha=0.4)
        else:
            ax_right.fill_between([rolling_dates[i], rolling_dates[i+1]], 
                                  rolling_accuracy[i], 0.5, color="#FADADD", alpha=0.4)
    
    # Formatting
    ax_right.set_xlabel("Date", fontsize=11)
    ax_right.set_ylabel("Directional Accuracy", fontsize=11)
    ax_right.set_title("Rolling 12-Month Directional Accuracy", 
                       fontsize=12, fontweight="bold")
    ax_right.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    ax_right.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.setp(ax_right.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax_right.set_ylim(0.4, 0.75)
    ax_right.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax_right.grid(True, alpha=0.2, axis="y")
    
    plt.tight_layout()
    fig.savefig(figures_dir / "fig2_prediction_quality.png",
                bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print("  ✓ fig2_prediction_quality.png")


def fig3_garch_volatility(sector_vol_forecasts, figures_dir):
    """Figure 3: GARCH volatility forecasts over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if sector_vol_forecasts is None or sector_vol_forecasts.empty:
        print("  ⚠ Skipping fig3_garch_volatility.png (no data)")
        plt.close(fig)
        return
    
    # Extract three sectors: XLE, XLP, XLK
    sectors_to_plot = ["XLE", "XLP", "XLK"]
    colors_map = {"XLE": RED, "XLP": GREEN, "XLK": MID_BLUE}
    labels_map = {"XLE": "XLE (Energy)", "XLP": "XLP (Consumer Staples)", 
                  "XLK": "XLK (Technology)"}
    
    for sector in sectors_to_plot:
        if sector in sector_vol_forecasts.columns:
            vol_pct = sector_vol_forecasts[sector].values * 100
            ax.plot(sector_vol_forecasts.index, vol_pct, 
                   color=colors_map[sector], linewidth=1.8, 
                   label=labels_map[sector])
    
    # COVID crash shading (March 2020 - June 2020)
    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2020-06-30")
    ax.axvspan(covid_start, covid_end, color="gray", alpha=0.15)
    ax.text(covid_start + pd.Timedelta(days=45), ax.get_ylim()[1] * 0.95,
            "COVID\nCrash", fontsize=10, ha="center", color="black", fontweight="bold")
    
    # Find and annotate XLE peak
    if "XLE" in sector_vol_forecasts.columns:
        xle_series = sector_vol_forecasts["XLE"]
        peak_idx = xle_series.idxmax()
        peak_val = xle_series.max() * 100
        ax.annotate(f"XLE peak\n{peak_val:.0f}%",
                   xy=(peak_idx, peak_val),
                   xytext=(peak_idx + pd.Timedelta(days=60), peak_val + 5),
                   color=RED, fontsize=9, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    
    # Formatting
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
    ax.set_title("GARCH(1,1) Sector Volatility Forecasts — Daily Returns",
                fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(symbol="%"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2, axis="y")
    
    plt.tight_layout()
    fig.savefig(figures_dir / "fig3_garch_volatility.png",
                bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print("  ✓ fig3_garch_volatility.png")


def fig4_portfolio_weights(forecast_backtest, figures_dir):
    """Figure 4: Portfolio weights over time (stacked area)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract weights from trade log
    trade_log = forecast_backtest.get("trade_log_df")
    if trade_log is None or trade_log.empty:
        print("  ⚠ Skipping fig4_portfolio_weights.png (no trade log)")
        plt.close(fig)
        return
    
    # Get sector columns (should be the numeric columns with weights)
    from simple_strategy.config import SECTOR_TICKERS
    
    # Filter to only sector columns that exist
    sector_cols = [s for s in SECTOR_TICKERS if s in trade_log.columns]
    
    if len(sector_cols) == 0:
        print("  ⚠ Skipping fig4_portfolio_weights.png (no sector weight columns)")
        plt.close(fig)
        return
    
    # Sort by date if not already
    weights_df = trade_log[sector_cols].copy()
    if not isinstance(weights_df.index, pd.DatetimeIndex):
        weights_df.index = pd.to_datetime(weights_df.index)
    weights_df = weights_df.sort_index()
    
    # Generate color palette
    cmap = plt.cm.get_cmap("tab20")
    colors = [cmap(i / len(sector_cols)) for i in range(len(sector_cols))]
    
    # Stacked area plot
    ax.stackplot(weights_df.index, 
                [weights_df[s].values for s in sector_cols],
                labels=sector_cols, colors=colors, alpha=0.85)
    
    # Formatting
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Weight", fontsize=11)
    ax.set_title("Monthly Portfolio Weights by Sector", 
                fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # Legend outside plot
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
             fontsize=8, ncol=1, framealpha=0.95)
    
    plt.tight_layout()
    fig.savefig(figures_dir / "fig4_portfolio_weights.png",
                bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print("  ✓ fig4_portfolio_weights.png")


def fig5_return_distribution(monthly_results_df, figures_dir):
    """Figure 5: Monthly return distribution comparison."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
    
    if monthly_results_df is None or monthly_results_df.empty:
        print("  ⚠ Skipping fig5_return_distribution.png (no data)")
        plt.close(fig)
        return
    
    strategy_returns = monthly_results_df.get("strategy_return", pd.Series())
    baseline_returns = monthly_results_df.get("baseline_return", pd.Series())
    
    if strategy_returns.empty or baseline_returns.empty:
        print("  ⚠ Skipping fig5_return_distribution.png (missing columns)")
        plt.close(fig)
        return
    
    # Panel A: Overlapping histograms
    ax_left.hist(strategy_returns, bins=25, color=MID_BLUE, alpha=0.55,
                edgecolor="white", linewidth=0.4, label="Strategy")
    ax_left.hist(baseline_returns, bins=25, color=ORANGE, alpha=0.55,
                edgecolor="white", linewidth=0.4, label="Baseline")
    
    # Add mean lines
    strat_mean = strategy_returns.mean()
    base_mean = baseline_returns.mean()
    ax_left.axvline(strat_mean, color=MID_BLUE, linestyle="--", linewidth=1.2)
    ax_left.axvline(base_mean, color=ORANGE, linestyle="--", linewidth=1.2)
    
    # Annotations
    strat_std = strategy_returns.std()
    base_std = baseline_returns.std()
    ann_text = f"Strategy: μ={strat_mean:.2%}  σ={strat_std:.2%}\nBaseline: μ={base_mean:.2%}  σ={base_std:.2%}"
    ax_left.text(0.05, 0.95, ann_text, transform=ax_left.transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    
    ax_left.set_xlabel("Monthly Return", fontsize=11)
    ax_left.set_ylabel("Frequency", fontsize=11)
    ax_left.set_title("Monthly Return Distribution", fontsize=12, fontweight="bold")
    ax_left.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax_left.grid(True, alpha=0.2, axis="y")
    
    # Panel B: Box plot
    bp = ax_right.boxplot([strategy_returns, baseline_returns], 
                          patch_artist=True,
                          labels=["Strategy", "Baseline"],
                          widths=0.6)
    
    # Color boxes
    bp["boxes"][0].set_facecolor(LIGHT_BLUE)
    bp["boxes"][0].set_edgecolor(MID_BLUE)
    bp["boxes"][1].set_facecolor("#FAE5D3")
    bp["boxes"][1].set_edgecolor(ORANGE)
    
    # Color median lines
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
    
    # Add zero line
    ax_right.axhline(y=0, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.7)
    
    ax_right.set_ylabel("Monthly Return", fontsize=11)
    ax_right.set_title("Return Distribution — Box Plot", fontsize=12, fontweight="bold")
    ax_right.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax_right.grid(True, alpha=0.2, axis="y")
    
    plt.tight_layout()
    fig.savefig(figures_dir / "fig5_return_distribution.png",
                bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print("  ✓ fig5_return_distribution.png")


def fig6_feature_importance(feature_importance, figures_dir):
    """Figure 6: Feature importance."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    if feature_importance is None or feature_importance.empty:
        print("  ⚠ Skipping fig6_feature_importance.png (no feature importance data)")
        plt.close(fig)
        return
    
    # Sort by importance
    fi_sorted = feature_importance.sort_values("importance", ascending=True)
    
    # Clean up feature names
    feature_name_map = {
        "mom_1m": "Mom 1m",
        "mom_3m": "Mom 3m",
        "mom_6m": "Mom 6m",
        "mom_12m": "Mom 12m",
        "vol_3m": "Vol 3m",
        "vol_6m": "Vol 6m",
        "vol_12m": "Vol 12m",
        "rev_1m": "Rev 1m",
    }
    
    clean_names = [feature_name_map.get(f, f) for f in fi_sorted["feature"]]
    
    # Plot
    bars = ax.barh(range(len(fi_sorted)), fi_sorted["importance"], 
                  color=MID_BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    
    # Error bars
    ax.errorbar(fi_sorted["importance"], range(len(fi_sorted)),
               xerr=fi_sorted["std_dev"], fmt="none", ecolor=GRAY, 
               capsize=3, elinewidth=1, alpha=0.7)
    
    # Uniform baseline line
    n_features = len(fi_sorted)
    uniform_importance = 1.0 / n_features
    ax.axvline(uniform_importance, color=GRAY, linestyle="--", linewidth=1.0,
              label=f"Uniform (1/{n_features})", alpha=0.7)
    
    # Value labels
    for i, v in enumerate(fi_sorted["importance"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    
    # Formatting
    ax.set_yticks(range(len(clean_names)))
    ax.set_yticklabels(clean_names, fontsize=10)
    ax.set_xlabel("Mean Importance", fontsize=11)
    ax.set_title("Random Forest Feature Importance\n(Mean ± Std across Rolling Windows)",
                fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.2, axis="x")
    
    plt.tight_layout()
    fig.savefig(figures_dir / "fig6_feature_importance.png",
                bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print("  ✓ fig6_feature_importance.png")


def generate_all_figures(
    forecast_backtest,
    baseline_backtest,
    predictions,
    actual_returns,
    sector_vol_forecasts,
    feature_importance,
    monthly_results_df,
    figures_dir,
):
    """
    Generate all 6 figures and save to figures_dir.
    
    Parameters
    ----------
    forecast_backtest : dict
        Results from forecast-driven backtest
    baseline_backtest : dict
        Results from baseline momentum backtest
    predictions : pd.DataFrame
        ML model predictions (dates × sectors)
    actual_returns : pd.DataFrame
        Actual realized returns (dates × sectors)
    sector_vol_forecasts : pd.DataFrame
        GARCH volatility forecasts (dates × sectors)
    feature_importance : pd.DataFrame
        Feature importance with std deviations
    monthly_results_df : pd.DataFrame
        Monthly backtest results with strategy/baseline returns
    figures_dir : Path
        Directory to save all figures
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[STEP 9] Generating figures...")
    
    try:
        fig1_equity_curves(forecast_backtest, baseline_backtest, figures_dir)
    except Exception as e:
        print(f"  ⚠ Error generating fig1_equity_curves.png: {e}")
    
    try:
        fig2_prediction_quality(predictions, actual_returns, figures_dir)
    except Exception as e:
        print(f"  ⚠ Error generating fig2_prediction_quality.png: {e}")
    
    try:
        fig3_garch_volatility(sector_vol_forecasts, figures_dir)
    except Exception as e:
        print(f"  ⚠ Error generating fig3_garch_volatility.png: {e}")
    
    try:
        fig4_portfolio_weights(forecast_backtest, figures_dir)
    except Exception as e:
        print(f"  ⚠ Error generating fig4_portfolio_weights.png: {e}")
    
    try:
        fig5_return_distribution(monthly_results_df, figures_dir)
    except Exception as e:
        print(f"  ⚠ Error generating fig5_return_distribution.png: {e}")
    
    try:
        fig6_feature_importance(feature_importance, figures_dir)
    except Exception as e:
        print(f"  ⚠ Error generating fig6_feature_importance.png: {e}")
    
    print(f"✓ Figures saved to {figures_dir}")
