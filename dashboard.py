# ============================================================
#  Retail Sales Demand Forecasting – Streamlit Dashboard
#  B.Sc Data Science – Major Project
#  Authors: Ayush Bisen | Nishika Urkude
#  Run: streamlit run dashboard.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
BORDER  = "#2e3250"
TEXT    = "#c9d1e0"
ACCENT  = "#4f8ef7"
ACCENT2 = "#f7834f"
ACCENT3 = "#4ff7a0"
WARN    = "#f7d04f"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      "#7a85a3",
    "ytick.color":      "#7a85a3",
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

st.markdown(f"""
<style>
  /* ── Base ── */
  html, body, [class*="css"] {{ background-color: {BG}; color: {TEXT}; font-family: 'Segoe UI', sans-serif; }}
  .main {{ background-color: {BG}; }}
  section[data-testid="stSidebar"] {{ background-color: {PANEL}; border-right: 1px solid {BORDER}; }}

  /* ── Metric cards ── */
  [data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px 18px;
  }}
  [data-testid="stMetric"] label {{ color: #7a85a3 !important; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
  [data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-size: 26px; font-weight: 700; }}

  /* ── Header band ── */
  .header-band {{
    background: linear-gradient(135deg, #1a1d27 0%, #0f1117 60%, #141827 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
  }}
  .header-band h1 {{ color: {ACCENT}; margin:0; font-size: 2rem; }}
  .header-band p  {{ color: #7a85a3; margin:4px 0 0; font-size: 0.9rem; }}

  /* ── Section labels ── */
  .section-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4f5a7a;
    margin-bottom: 8px;
  }}

  /* ── Tabs ── */
  [data-testid="stHorizontalBlock"] {{ gap: 12px; }}
  button[data-baseweb="tab"] {{ color: #7a85a3 !important; }}
  button[data-baseweb="tab"][aria-selected="true"] {{ color: {ACCENT} !important; border-bottom-color: {ACCENT} !important; }}

  /* ── Divider ── */
  hr {{ border-color: {BORDER}; margin: 18px 0; }}

  /* ── Sidebar widgets ── */
  [data-testid="stSelectbox"] label,
  [data-testid="stSlider"] label,
  [data-testid="stMultiselect"] label {{ color: #7a85a3; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
  div[data-baseweb="select"] > div {{ background: {PANEL} !important; border-color: {BORDER} !important; color: {TEXT} !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def make_fig():
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    return fig, ax

def holt_smoothing(series, alpha=0.3, beta=0.1, steps=12):
    s = [series.iloc[0]]
    b = [series.iloc[1] - series.iloc[0]]
    for val in series.iloc[1:]:
        s_prev, b_prev = s[-1], b[-1]
        s.append(alpha * val + (1 - alpha) * (s_prev + b_prev))
        b.append(beta * (s[-1] - s_prev) + (1 - beta) * b_prev)
    forecast = [s[-1] + i * b[-1] for i in range(1, steps + 1)]
    fitted   = [s[i] + b[i] for i in range(len(series))]
    return pd.Series(fitted, index=series.index), forecast

def arima_proxy(series, steps=12):
    diff = series.diff().dropna()
    x = diff.values[:-1]; y = diff.values[1:]
    beta = np.dot(x, y) / (np.dot(x, x) + 1e-9)
    intercept = diff.mean() * (1 - beta)
    curr = diff.iloc[-1]; last = series.iloc[-1]
    out = []
    for _ in range(steps):
        curr = intercept + beta * curr
        out.append(last + curr)
        last = out[-1]
    return out

def metrics(actual, predicted):
    a, p = np.array(actual), np.array(predicted)
    mae  = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p) ** 2))
    mape = np.mean(np.abs((a - p) / (a + 1e-9))) * 100
    return round(mae, 1), round(rmse, 1), round(mape, 2)

@st.cache_data
def load_data(file=None):
    if file:
        df = pd.read_csv(file, parse_dates=["Date"])
    else:
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range("2021-01-01", "2023-12-31", freq="W")
        n = len(dates)
        trend      = np.linspace(500, 900, n)
        seasonal   = 150 * np.sin(2 * np.pi * np.arange(n) / 52)
        noise      = np.random.normal(0, 40, n)
        sales      = np.clip(trend + seasonal + noise, 100, None)
        cats       = np.random.choice(["Electronics","Clothing","Groceries","Home & Garden","Sports"], n)
        regions    = np.random.choice(["North","South","East","West"], n)
        promo      = np.random.choice([0, 1], n, p=[0.75, 0.25])
        df = pd.DataFrame({
            "Date":       dates,
            "Sales":      np.round(sales, 2),
            "Units_Sold": np.round(sales / np.random.uniform(5,15,n)).astype(int),
            "Category":   cats,
            "Region":     regions,
            "Promotion":  promo,
            "Revenue":    np.round(sales * np.random.uniform(1.5, 2.5, n), 2),
        })
    return df.sort_values("Date").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='color:{ACCENT};font-weight:700;font-size:1.1rem;margin-bottom:16px'>⚙️ Controls</div>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                 help="Must have: Date, Sales, Category, Region")
    df_raw = load_data(uploaded)

    categories = ["All"] + sorted(df_raw["Category"].unique().tolist())
    regions    = ["All"] + sorted(df_raw["Region"].unique().tolist())
    sel_cat    = st.selectbox("Category", categories)
    sel_reg    = st.selectbox("Region",   regions)

    date_min = df_raw["Date"].min().date()
    date_max = df_raw["Date"].max().date()
    date_range = st.date_input("Date Range", value=(date_min, date_max),
                                min_value=date_min, max_value=date_max)

    st.markdown("---")
    st.markdown(f"<div class='section-label'>Forecasting</div>", unsafe_allow_html=True)
    model_choice = st.radio("Model", ["Holt's Exponential Smoothing", "ARIMA-like (AR1)", "Both"])
    forecast_wks = st.slider("Forecast Weeks", 4, 52, 12, step=4)
    alpha_val    = st.slider("Smoothing α", 0.05, 0.95, 0.35)
    beta_val     = st.slider("Trend β",     0.01, 0.50, 0.08)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:11px;color:#4f5a7a;line-height:1.6'>
      <b style='color:#7a85a3'>Major Project</b><br>
      B.Sc Data Science – Sem VI<br>
      Kamla Nehru Mahavidyalaya<br><br>
      <b style='color:#7a85a3'>Team</b><br>
      Ayush Bisen · Nishika Urkude
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_cat != "All": df = df[df["Category"] == sel_cat]
if sel_reg != "All": df = df[df["Region"]   == sel_reg]
if len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df[(df["Date"] >= d0) & (df["Date"] <= d1)]

series = df.groupby("Date")["Sales"].sum()
train_size = int(len(series) * 0.8)
train = series.iloc[:train_size]
test  = series.iloc[train_size:]

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-band">
  <h1>📊 Retail Sales Forecasting Dashboard</h1>
  <p>Demand Analysis & Business Decision Support &nbsp;·&nbsp; {sel_cat} &nbsp;·&nbsp; {sel_reg}</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total_sales = df["Sales"].sum()
total_rev   = df["Revenue"].sum()
avg_weekly  = series.mean()
peak_week   = series.max()
promo_lift  = df[df.Promotion==1]["Sales"].mean() - df[df.Promotion==0]["Sales"].mean()

k1.metric("Total Sales",    f"₹{total_sales:,.0f}")
k2.metric("Total Revenue",  f"₹{total_rev:,.0f}")
k3.metric("Avg Weekly",     f"₹{avg_weekly:,.0f}")
k4.metric("Peak Week",      f"₹{peak_week:,.0f}")
k5.metric("Promo Lift",     f"₹{promo_lift:+,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Sales Trends", "🔬 EDA", "🤖 Forecasting", "📋 Model Metrics"])

# ══════════════════════════════════════════
# TAB 1 – SALES TRENDS
# ══════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig, ax = make_fig()
        ax.plot(series.index, series.values, color=ACCENT, linewidth=2, alpha=0.9)
        ax.fill_between(series.index, series.values, alpha=0.12, color=ACCENT)
        # Rolling average
        roll = series.rolling(4).mean()
        ax.plot(roll.index, roll.values, color=ACCENT2, linewidth=1.5,
                linestyle="--", alpha=0.8, label="4-wk Rolling Avg")
        ax.set_title("Weekly Sales with Rolling Average", color="white", pad=12)
        ax.set_xlabel("Date"); ax.set_ylabel("Sales (₹)")
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close()

    with col_b:
        fig, ax = make_fig()
        fig.set_size_inches(5, 4)
        month_avg = df.copy()
        month_avg["Month"] = month_avg["Date"].dt.strftime("%b")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        mavg = month_avg.groupby("Month")["Sales"].mean().reindex(month_order).dropna()
        bars = ax.bar(mavg.index, mavg.values,
                      color=[ACCENT if v >= mavg.mean() else ACCENT2 for v in mavg.values],
                      edgecolor=BORDER, linewidth=0.5, alpha=0.85)
        ax.axhline(mavg.mean(), color=WARN, linewidth=1.2, linestyle="--", label="Mean")
        ax.set_title("Monthly Average Sales", color="white", pad=10)
        ax.set_ylabel("Avg Sales (₹)")
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════
# TAB 2 – EDA
# ══════════════════════════════════════════
with tab2:
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig, ax = make_fig()
        palette = [ACCENT, ACCENT2, ACCENT3, WARN, "#c44ff7"]
        cats = df["Category"].unique()
        for i, cat in enumerate(cats):
            vals = df[df.Category == cat]["Sales"]
            ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=palette[i % len(palette)], alpha=0.75),
                       medianprops=dict(color="white", linewidth=2),
                       whiskerprops=dict(color=palette[i % len(palette)]),
                       capprops=dict(color=palette[i % len(palette)]),
                       flierprops=dict(marker="o", color=palette[i % len(palette)],
                                       alpha=0.4, markersize=3))
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
        ax.set_title("Sales Distribution by Category", color="white")
        ax.set_ylabel("Sales (₹)")
        st.pyplot(fig); plt.close()

    with r1c2:
        fig, ax = make_fig()
        region_avg = df.groupby("Region")["Sales"].mean().sort_values()
        colors_r = [ACCENT, ACCENT2, ACCENT3, WARN]
        bars = ax.barh(region_avg.index, region_avg.values,
                       color=colors_r[:len(region_avg)], alpha=0.85)
        for bar, val in zip(bars, region_avg.values):
            ax.text(val + 3, bar.get_y() + bar.get_height()/2,
                    f"₹{val:.0f}", va="center", fontsize=10, color=TEXT)
        ax.set_title("Avg Sales by Region", color="white")
        ax.set_xlabel("Avg Sales (₹)")
        st.pyplot(fig); plt.close()

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig, ax = make_fig()
        df2 = df.copy()
        df2["Year"]  = df2["Date"].dt.year
        df2["Month"] = df2["Date"].dt.month
        pivot = df2.pivot_table(values="Sales", index="Year", columns="Month", aggfunc="mean")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [month_names[m-1] for m in pivot.columns]
        sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt=".0f",
                    linewidths=0.5, linecolor=BG,
                    cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
        ax.set_title("Monthly Sales Heatmap", color="white")
        st.pyplot(fig); plt.close()

    with r2c2:
        fig, ax = make_fig()
        promo_data = df.groupby("Promotion")["Sales"].mean()
        labels = ["No Promotion", "With Promotion"]
        colors_p = [ACCENT2, ACCENT3]
        wedges, texts, autotexts = ax.pie(
            promo_data.values, labels=labels, colors=colors_p,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": TEXT, "fontsize": 10},
            wedgeprops={"edgecolor": BG, "linewidth": 2})
        for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
        ax.set_title("Sales Share: Promotion Effect", color="white")
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════
# TAB 3 – FORECASTING
# ══════════════════════════════════════════
with tab3:
    future_dates = pd.date_range(
        start=series.index[-1] + pd.Timedelta(weeks=1),
        periods=forecast_wks, freq="W")

    holt_fitted, holt_forecast = holt_smoothing(train, alpha_val, beta_val, len(test)+forecast_wks)
    holt_test  = pd.Series(holt_forecast[:len(test)], index=test.index)
    holt_fut   = holt_forecast[len(test):]
    arima_fut_all = arima_proxy(train, steps=len(test)+forecast_wks)
    arima_test = pd.Series(arima_fut_all[:len(test)], index=test.index)
    arima_fut  = arima_fut_all[len(test):]

    models_to_show = []
    if model_choice in ["Holt's Exponential Smoothing", "Both"]:
        models_to_show.append(("Holt's Exponential Smoothing", holt_test, holt_fut, ACCENT2))
    if model_choice in ["ARIMA-like (AR1)", "Both"]:
        models_to_show.append(("ARIMA-like Forecast (AR1)", arima_test, arima_fut, ACCENT3))

    for (name, test_pred, fut_vals, color) in models_to_show:
        fig, ax = plt.subplots(figsize=(13, 4.5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)

        ax.plot(train.index, train.values, color=ACCENT,  linewidth=1.8, label="Train",        alpha=0.9)
        ax.plot(test.index,  test.values,  color="white", linewidth=1.5, label="Actual (Test)", alpha=0.9)
        ax.plot(test.index,  test_pred.values, color=color, linewidth=1.8,
                linestyle="--", label="Predicted (Test)", alpha=0.9)
        ax.plot(future_dates, fut_vals, color=color, linewidth=2.2,
                linestyle=":", label=f"Forecast (+{forecast_wks}wk)", alpha=0.9)
        ax.fill_between(future_dates,
                        np.array(fut_vals) * 0.87,
                        np.array(fut_vals) * 1.13,
                        alpha=0.12, color=color, label="±13% Band")

        mae, rmse, mape = metrics(test.values, test_pred.values)
        ax.set_title(f"{name}   |   MAE={mae}  RMSE={rmse}  MAPE={mape}%",
                     color="white", pad=10, fontsize=12)
        ax.set_ylabel("Sales (₹)")
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        st.pyplot(fig); plt.close()

    # -- Forecast table
    st.markdown(f"<div class='section-label' style='margin-top:12px'>Forecast Values</div>",
                unsafe_allow_html=True)
    tbl = pd.DataFrame({"Week": future_dates.strftime("%d %b %Y")})
    if model_choice in ["Holt's Exponential Smoothing", "Both"]:
        tbl["Holt Forecast (₹)"] = [f"{v:,.0f}" for v in holt_fut]
    if model_choice in ["ARIMA-like (AR1)", "Both"]:
        tbl["ARIMA Forecast (₹)"] = [f"{v:,.0f}" for v in arima_fut]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════
# TAB 4 – MODEL METRICS
# ══════════════════════════════════════════
with tab4:
    holt_fitted2, holt_fc2 = holt_smoothing(train, alpha_val, beta_val, len(test))
    ht_pred = pd.Series(holt_fc2[:len(test)], index=test.index)
    at_pred = pd.Series(arima_proxy(train, steps=len(test))[:len(test)], index=test.index)

    h_mae, h_rmse, h_mape = metrics(test.values, ht_pred.values)
    a_mae, a_rmse, a_mape = metrics(test.values, at_pred.values)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"<div style='color:{ACCENT2};font-weight:700;margin-bottom:8px'>Holt's Exponential Smoothing</div>",
                    unsafe_allow_html=True)
        st.metric("MAE",    f"₹{h_mae:,}")
        st.metric("RMSE",   f"₹{h_rmse:,}")
        st.metric("MAPE",   f"{h_mape}%")
    with mc2:
        st.markdown(f"<div style='color:{ACCENT3};font-weight:700;margin-bottom:8px'>ARIMA-like (AR1 Differenced)</div>",
                    unsafe_allow_html=True)
        st.metric("MAE",    f"₹{a_mae:,}")
        st.metric("RMSE",   f"₹{a_rmse:,}")
        st.metric("MAPE",   f"{a_mape}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Residuals
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Residual Analysis", color="white", fontsize=13)
    for ax, pred, color, name in zip(
            axes,
            [ht_pred, at_pred],
            [ACCENT2, ACCENT3],
            ["Holt's", "ARIMA-like"]):
        ax.set_facecolor(PANEL)
        residuals = test.values - pred.values
        ax.bar(range(len(residuals)), residuals, color=color, alpha=0.7, edgecolor=BORDER)
        ax.axhline(0, color="white", linewidth=1.2, linestyle="--")
        ax.axhline(residuals.mean(), color=WARN, linewidth=1, linestyle=":",
                   label=f"Mean={residuals.mean():.1f}")
        ax.set_title(f"{name} Residuals", color="white")
        ax.set_xlabel("Test Week"); ax.set_ylabel("Residual (₹)")
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Actual vs Predicted scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(BG)
    for ax, pred, color, name in zip(
            axes,
            [ht_pred, at_pred],
            [ACCENT2, ACCENT3],
            ["Holt's", "ARIMA-like"]):
        ax.set_facecolor(PANEL)
        ax.scatter(test.values, pred.values, color=color, alpha=0.7, edgecolors=BORDER, s=50)
        lims = [min(test.min(), pred.min()), max(test.max(), pred.max())]
        ax.plot(lims, lims, color="white", linewidth=1.2, linestyle="--", label="Perfect Fit")
        ax.set_title(f"{name}: Actual vs Predicted", color="white")
        ax.set_xlabel("Actual Sales (₹)"); ax.set_ylabel("Predicted (₹)")
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
