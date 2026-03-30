# ============================================================
#  Retail Sales Demand Forecasting
#  B.Sc Data Science – Major Project
#  Authors: Ayush Bisen | Nishika Urkude
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

# -----------------------------------------------------------
# 0. STYLE SETUP
# -----------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#2e3250",
    "axes.labelcolor":  "#c9d1e0",
    "xtick.color":      "#7a85a3",
    "ytick.color":      "#7a85a3",
    "text.color":       "#c9d1e0",
    "grid.color":       "#2e3250",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.labelsize":   11,
})

ACCENT   = "#4f8ef7"
ACCENT2  = "#f7834f"
ACCENT3  = "#4ff7a0"
WARN     = "#f7d04f"

# -----------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------
def load_data(path="retail_sales_data.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# -----------------------------------------------------------
# 2. EDA HELPERS
# -----------------------------------------------------------
def eda_summary(df):
    print("=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)
    print(f"\nShape      : {df.shape}")
    print(f"Date Range : {df.Date.min().date()} → {df.Date.max().date()}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDescriptive Stats:\n{df[['Sales','Units_Sold','Revenue']].describe().round(2)}")

def plot_eda(df, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Retail Sales – Exploratory Data Analysis", fontsize=18,
                 fontweight="bold", color="white", y=0.98)
    fig.patch.set_facecolor("#0f1117")

    # -- 1. Sales over time
    ax = axes[0, 0]
    ax.plot(df.Date, df.Sales, color=ACCENT, linewidth=1.5, alpha=0.9)
    ax.fill_between(df.Date, df.Sales, alpha=0.15, color=ACCENT)
    ax.set_title("Weekly Sales Trend")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales (₹)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    # -- 2. Sales by Category (box)
    ax = axes[0, 1]
    cats = df["Category"].unique()
    palette = [ACCENT, ACCENT2, ACCENT3, WARN, "#c44ff7"]
    for i, cat in enumerate(cats):
        vals = df[df.Category == cat]["Sales"]
        bp = ax.boxplot(vals, positions=[i], widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=palette[i % len(palette)], alpha=0.7),
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color=palette[i % len(palette)]),
                        capprops=dict(color=palette[i % len(palette)]),
                        flierprops=dict(marker="o", color=palette[i % len(palette)], alpha=0.4, markersize=4))
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_title("Sales Distribution by Category")
    ax.set_ylabel("Sales (₹)")

    # -- 3. Monthly average sales heatmap by year
    ax = axes[1, 0]
    df["Year"]  = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    pivot = df.pivot_table(values="Sales", index="Year", columns="Month", aggfunc="mean")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m-1] for m in pivot.columns]
    sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt=".0f",
                linewidths=0.5, linecolor="#0f1117",
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    ax.set_title("Avg Monthly Sales Heatmap")
    ax.set_xlabel("Month"); ax.set_ylabel("Year")

    # -- 4. Sales by Region
    ax = axes[1, 1]
    region_avg = df.groupby("Region")["Sales"].mean().sort_values(ascending=True)
    bars = ax.barh(region_avg.index, region_avg.values, color=[ACCENT, ACCENT2, ACCENT3, WARN])
    for bar, val in zip(bars, region_avg.values):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                f"₹{val:.0f}", va="center", fontsize=10, color="white")
    ax.set_title("Average Sales by Region")
    ax.set_xlabel("Avg Sales (₹)")

    plt.tight_layout()
    if save:
        plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
        print("[✓] Saved: eda_plots.png")
    plt.show()

# -----------------------------------------------------------
# 3. SIMPLE FORECASTING (without statsmodels dependency)
#    Implements SES and a manual ARIMA-like approach
# -----------------------------------------------------------

def simple_exponential_smoothing(series, alpha=0.3):
    """Holt's Simple Exponential Smoothing (manual)."""
    result = [series.iloc[0]]
    for val in series.iloc[1:]:
        result.append(alpha * val + (1 - alpha) * result[-1])
    return pd.Series(result, index=series.index)

def holt_smoothing(series, alpha=0.3, beta=0.1, steps=12):
    """Holt's Double Exponential Smoothing with forecast."""
    s = [series.iloc[0]]
    b = [series.iloc[1] - series.iloc[0]]
    for val in series.iloc[1:]:
        s_prev, b_prev = s[-1], b[-1]
        s.append(alpha * val + (1 - alpha) * (s_prev + b_prev))
        b.append(beta * (s[-1] - s_prev) + (1 - beta) * b_prev)
    # forecast
    forecast = [s[-1] + i * b[-1] for i in range(1, steps + 1)]
    fitted = [s[i] + b[i] for i in range(len(series))]
    return pd.Series(fitted, index=series.index), forecast

def differencing_forecast(series, steps=12):
    """Simple AR(1) on differenced series – approximates ARIMA(1,1,0)."""
    diff = series.diff().dropna()
    # Fit AR(1) coefficient via OLS
    y = diff.values[1:]
    x = diff.values[:-1]
    beta = np.dot(x, y) / np.dot(x, x)
    intercept = diff.mean() * (1 - beta)
    last_diff = diff.iloc[-1]
    last_val  = series.iloc[-1]
    forecasted_diffs = []
    curr_diff = last_diff
    for _ in range(steps):
        curr_diff = intercept + beta * curr_diff + np.random.normal(0, diff.std() * 0.1)
        forecasted_diffs.append(curr_diff)
    forecast_vals = [last_val]
    for d in forecasted_diffs:
        forecast_vals.append(forecast_vals[-1] + d)
    return forecast_vals[1:]

def evaluate(actual, predicted):
    mae  = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2)}

# -----------------------------------------------------------
# 4. MAIN FORECASTING PIPELINE
# -----------------------------------------------------------
def run_forecasting(df, steps=12, save=True):
    series = df.groupby("Date")["Sales"].sum()
    train_size = int(len(series) * 0.8)
    train = series.iloc[:train_size]
    test  = series.iloc[train_size:]

    # ---- Holt's (Exponential Smoothing) ----
    holt_fitted, holt_forecast = holt_smoothing(train, alpha=0.35, beta=0.08, steps=len(test)+steps)
    holt_test_pred = pd.Series(holt_forecast[:len(test)], index=test.index)
    holt_future    = holt_forecast[len(test):]

    # ---- AR(1) on differenced series (ARIMA proxy) ----
    arima_test_pred_vals = differencing_forecast(train, steps=len(test)+steps)
    arima_test_pred = pd.Series(arima_test_pred_vals[:len(test)], index=test.index)
    arima_future    = arima_test_pred_vals[len(test):]

    # ---- Metrics ----
    holt_metrics  = evaluate(test.values, holt_test_pred.values)
    arima_metrics = evaluate(test.values, arima_test_pred.values)

    print("\n" + "=" * 55)
    print("  MODEL EVALUATION")
    print("=" * 55)
    print(f"\n  Exponential Smoothing (Holt's):")
    for k, v in holt_metrics.items():
        print(f"    {k:8s}: {v}")
    print(f"\n  ARIMA-like (AR1 on diff series):")
    for k, v in arima_metrics.items():
        print(f"    {k:8s}: {v}")

    # ---- Future dates ----
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                 periods=steps, freq="W")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    fig.suptitle("Retail Sales Demand Forecasting", fontsize=18,
                 fontweight="bold", color="white", y=0.99)
    fig.patch.set_facecolor("#0f1117")

    for ax, (model_name, test_pred, future_vals, color) in zip(
            axes,
            [("Holt's Exponential Smoothing", holt_test_pred, holt_future, ACCENT2),
             ("ARIMA-like (AR1 Differenced)", arima_test_pred, arima_future, ACCENT3)]):

        ax.plot(train.index, train.values, color=ACCENT, linewidth=1.8, label="Train", alpha=0.9)
        ax.plot(test.index,  test.values,  color="white", linewidth=1.5, label="Actual (Test)", alpha=0.9)
        ax.plot(test.index,  test_pred.values,  color=color, linewidth=1.8,
                linestyle="--", label="Forecast (Test)", alpha=0.9)
        ax.plot(future_dates, future_vals, color=color, linewidth=2,
                linestyle=":", label=f"Future {steps}wk Forecast", alpha=0.85)
        ax.fill_between(future_dates,
                        np.array(future_vals) * 0.88,
                        np.array(future_vals) * 1.12,
                        alpha=0.12, color=color, label="Confidence Band")

        m = evaluate(test.values, test_pred.values)
        ax.set_title(f"{model_name}   |   MAE={m['MAE']}  RMSE={m['RMSE']}  MAPE={m['MAPE%']}%",
                     fontsize=12, color="white")
        ax.set_ylabel("Sales (₹)")
        ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#2e3250", labelcolor="white")
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    if save:
        plt.savefig("forecast_plots.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
        print("[✓] Saved: forecast_plots.png")
    plt.show()

    return {
        "series": series,
        "train":  train,
        "test":   test,
        "holt_metrics":  holt_metrics,
        "arima_metrics": arima_metrics,
        "future_dates":  future_dates,
        "holt_future":   holt_future,
        "arima_future":  arima_future,
    }

# -----------------------------------------------------------
# 5. ENTRY POINT
# -----------------------------------------------------------
if __name__ == "__main__":
    df = load_data("retail_sales_data.csv")
    eda_summary(df)
    plot_eda(df)
    run_forecasting(df, steps=12)
    print("\n[✓] Analysis complete.")
