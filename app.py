import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Semiconductor Wafer Cost Intelligence", layout="wide")

st.title("Semiconductor Wafer Cost Intelligence Platform")
st.caption("Polysilicon → Wafer → Die → Margin | Country & Policy Intelligence")

df = pd.read_csv("data/polysilicon_daily_synthetic_2016_2025.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

forecast = pd.read_csv("data/wafer_material_cost_forecast_2026.csv")

np.random.seed(42)
df["dollar_index"] = 100 - 0.2 * df["price_usd_per_kg"] + np.random.normal(0, 0.5, len(df))
df["crude_oil"] = 60 + 0.8 * df["price_usd_per_kg"] + np.random.normal(0, 3, len(df))
df["copper"] = 6000 + 120 * df["price_usd_per_kg"] + np.random.normal(0, 80, len(df))
df["gold"] = 1400 + 5 * df["price_usd_per_kg"] + np.random.normal(0, 15, len(df))
df["silver"] = 18 + 0.08 * df["price_usd_per_kg"] + np.random.normal(0, 0.5, len(df))

st.sidebar.header("Controls")
lag = st.sidebar.slider("Supply Chain Lag (days)", 0, 365, 180)
shock = st.sidebar.slider("Polysilicon Price Shock (%)", -30, 50, 10)

node_map = {"65 nm": 700, "28 nm": 900, "14 nm": 1200, "7 nm": 1600}
node = st.sidebar.selectbox("Technology Node", list(node_map.keys()))
dies_per_wafer = node_map[node]

tabs = st.tabs(["Price Trend","Ratio Candlestick","Lagged Correlation","2026 Forecast","Shock → Margin","SHAP Explainability"])

with tabs[0]:
    st.line_chart(df.set_index("date")["price_usd_per_kg"])

with tabs[1]:
    df["ratio"] = df["price_usd_per_kg"] / (df["copper"] / 1000)
    ohlc = df.resample("M", on="date").agg({"ratio": ["first","max","min","last"]})
    ohlc.columns = ["open","high","low","close"]
    fig = go.Figure(data=[go.Candlestick(x=ohlc.index,open=ohlc["open"],high=ohlc["high"],low=ohlc["low"],close=ohlc["close"])])
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    corr = df["price_usd_per_kg"].corr(df["copper"].shift(lag))
    st.metric("Polysilicon–Copper Correlation", round(corr, 3))

with tabs[3]:
    st.line_chart(forecast.set_index("date")["forecast_wafer_material_cost_usd"])

with tabs[4]:
    shocked_price = df["price_usd_per_kg"] * (1 + shock/100)
    wafer_cost = shocked_price * 0.60 / 0.68
    die_cost = wafer_cost / dies_per_wafer
    ASP = 12.0
    margin = (ASP - die_cost) / ASP * 100
    st.metric("Avg Die Cost (USD)", round(die_cost.mean(), 4))
    st.metric("Gross Margin (%)", round(margin.mean(), 2))

with tabs[5]:
    df["poly_lag_180"] = df["price_usd_per_kg"].shift(180)
    df2 = df.dropna()
    X = df2[["poly_lag_180","crude_oil","dollar_index","copper","gold","silver"]]
    y = df2["price_usd_per_kg"] * 0.60 / 0.68
    model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(plt.gcf())
