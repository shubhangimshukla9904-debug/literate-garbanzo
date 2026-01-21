import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Semiconductor Wafer Cost Intelligence",
    layout="wide"
)

st.title("Semiconductor Wafer Cost Intelligence Platform")
st.caption("Polysilicon → Wafer → Die → Margin | Macro & Policy Intelligence")

# ===============================
# SAFE DATA LOADER (FIXES FILE ERROR)
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "polysilicon_daily_synthetic_2016_2025.csv")

@st.cache_data
def load_forecast():
    return pd.read_csv(DATA_DIR / "wafer_material_cost_forecast_2026.csv")

df = load_data()
forecast = load_forecast()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ===============================
# SYNTHETIC MACRO + FINANCIAL DATA
# ===============================
np.random.seed(42)

df["DXY"] = 100 - 0.25 * df["price_usd_per_kg"] + np.random.normal(0, 0.6, len(df))
df["Copper"] = 6000 + 120 * df["price_usd_per_kg"] + np.random.normal(0, 80, len(df))
df["Gold"] = 1400 + 4.5 * df["price_usd_per_kg"] + np.random.normal(0, 15, len(df))
df["Silver"] = 18 + 0.09 * df["price_usd_per_kg"] + np.random.normal(0, 0.6, len(df))

# VIX (risk index)
df["VIX"] = 18 + 0.4 * df["price_usd_per_kg"] + np.random.normal(0, 2, len(df))

# Industrial demand index (semiconductor + manufacturing proxy)
df["Industrial_Demand"] = 100 + 0.03 * df["Copper"] + np.random.normal(0, 5, len(df))

# GDP Growth (macro cycle proxy)
df["GDP_Growth"] = 2.5 + 0.002 * df["Industrial_Demand"] - 0.05 * df["VIX"]

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("Controls")

lag = st.sidebar.slider("Supply Chain Lag (days)", 0, 365, 180)
shock = st.sidebar.slider("Polysilicon Price Shock (%)", -30, 50, 10)

node_map = {"65 nm": 700, "28 nm": 900, "14 nm": 1200, "7 nm": 1600}
node = st.sidebar.selectbox("Technology Node", list(node_map.keys()))
dies_per_wafer = node_map[node]

# ===============================
# TABS
# ===============================
tabs = st.tabs([
    "📈 Price Trends",
    "🕯 Ratio Candlestick",
    "🔗 Macro Correlations",
    "⏱ Lagged Correlation",
    "🌍 Macro Cycle View",
    "⚠ Shock → Margin"
])

# ===============================
# TAB 1 – PRICE TREND
# ===============================
with tabs[0]:
    st.subheader("Polysilicon Price Trend")
    st.line_chart(df.set_index("date")["price_usd_per_kg"])

# ===============================
# TAB 2 – RATIO CANDLESTICK
# ===============================
with tabs[1]:
    st.subheader("Polysilicon / Copper Ratio")

    df["ratio"] = df["price_usd_per_kg"] / (df["Copper"] / 1000)
    ohlc = df.resample("M", on="date").agg({
        "ratio": ["first", "max", "min", "last"]
    })
    ohlc.columns = ["open", "high", "low", "close"]

    fig = go.Figure(data=[go.Candlestick(
        x=ohlc.index,
        open=ohlc["open"],
        high=ohlc["high"],
        low=ohlc["low"],
        close=ohlc["close"]
    )])

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# TAB 3 – CORRELATION MATRIX
# ===============================
with tabs[2]:
    st.subheader("Correlation: Macro & Metals")

    corr_cols = [
        "price_usd_per_kg",
        "DXY",
        "Copper",
        "Gold",
        "Silver",
        "VIX",
        "Industrial_Demand",
        "GDP_Growth"
    ]

    corr = df[corr_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# TAB 4 – LAGGED CORRELATION
# ===============================
with tabs[3]:
    st.subheader("Lagged Correlation (Polysilicon vs Copper)")

    lag_corr = df["price_usd_per_kg"].corr(df["Copper"].shift(lag))
    st.metric("Correlation Coefficient", round(lag_corr, 3))

# ===============================
# TAB 5 – MACRO CYCLE VIEW
# ===============================
with tabs[4]:
    st.subheader("Macro Cycle Indicators")

    st.line_chart(
        df.set_index("date")[[
            "VIX",
            "Industrial_Demand",
            "GDP_Growth"
        ]]
    )

# ===============================
# TAB 6 – SHOCK → DIE COST → MARGIN
# ===============================
with tabs[5]:
    st.subheader("Price Shock → Die Cost & Margin")

    shocked_price = df["price_usd_per_kg"] * (1 + shock / 100)
    wafer_cost = shocked_price * 0.60 / 0.68
    die_cost = wafer_cost / dies_per_wafer

    ASP = 12.0
    margin = (ASP - die_cost) / ASP * 100

    col1, col2 = st.columns(2)
    col1.metric("Avg Die Cost (USD)", round(die_cost.mean(), 4))
    col2.metric("Gross Margin (%)", round(margin.mean(), 2))
