#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SpendAnalyticsDashboard.py
# -----------------------------------------
# Modern Streamlit Spend Analytics Template
# -----------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------------
# Page Config (MUST be first Streamlit call)
# -----------------------------------------
st.set_page_config(
    page_title="Spend Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# Custom Styling (Dark, Professional)
# -----------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #2b2b2b;
    color: white;
}
h1, h2, h3, h4 {
    color: #f5f5f5;
}
[data-testid="metric-container"] {
    background-color: #3a3a3a;
    border-radius: 8px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Utility Functions
# -----------------------------------------
def safe_sum(df, col):
    """Safely sum a column"""
    return df[col].sum() if col in df.columns else 0

def to_cr(value):
    """Convert to Crores"""
    return value / 1e7

# -----------------------------------------
# Data Loading
# -----------------------------------------
@st.cache_data(ttl=600)
def load_data(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()
    df = pd.read_csv(uploaded_file)
    return df

# -----------------------------------------
# Sidebar
# -----------------------------------------
st.sidebar.title("📊 Spend Analytics")

uploaded_file = st.sidebar.file_uploader(
    "Upload Procurement CSV",
    type=["csv"]
)

df = load_data(uploaded_file)

if df.empty:
    st.info("⬅️ Upload a CSV file to begin analysis")
    st.stop()

# -----------------------------------------
# Filters
# -----------------------------------------
with st.sidebar.expander("Filters", expanded=True):
    supplier = st.multiselect(
        "Supplier",
        sorted(df["Supplier"].dropna().unique())
        if "Supplier" in df.columns else []
    )

if supplier:
    df = df[df["Supplier"].isin(supplier)]

# -----------------------------------------
# Header
# -----------------------------------------
st.title("📈 Spend Analytics & P2P Dashboard")
st.caption("Executive overview of procurement spend, savings & risks")

# -----------------------------------------
# KPI Row
# -----------------------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total POs",
    f"{len(df):,}"
)

k2.metric(
    "Actual Spend (₹ Cr)",
    f"{to_cr(safe_sum(df,'actual_spend')):,.2f}"
)

k3.metric(
    "Savings (₹ Cr)",
    f"{to_cr(safe_sum(df,'savings')):,.2f}"
)

late_pct = (
    (df["On_Time_Delivery"].str.lower() == "no").mean() * 100
    if "On_Time_Delivery" in df.columns else 0
)

k4.metric(
    "Late Delivery %",
    f"{late_pct:.1f}%"
)

# -----------------------------------------
# Spend Breakdown
# -----------------------------------------
st.markdown("## 💰 Spend Distribution")

c1, c2 = st.columns(2)

if "Item_Category" in df.columns:
    cat_spend = (
        df.groupby("Item_Category")["actual_spend"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_cat = px.bar(
        cat_spend,
        x="Item_Category",
        y="actual_spend",
        title="Top Categories by Spend",
        text_auto=".2s"
    )
    c1.plotly_chart(fig_cat, use_container_width=True)

if "Supplier" in df.columns:
    sup_spend = (
        df.groupby("Supplier")["actual_spend"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_sup = px.bar(
        sup_spend,
        x="Supplier",
        y="actual_spend",
        title="Top Suppliers by Spend",
        text_auto=".2s"
    )
    c2.plotly_chart(fig_sup, use_container_width=True)

# -----------------------------------------
# Exceptions & Risk
# -----------------------------------------
st.markdown("## 🚨 Exceptions & Risk Watchlist")

exceptions = []

if {"Unit_Price", "Negotiated_Price"}.issubset(df.columns):
    ex = df[df["Unit_Price"] > 1.15 * df["Negotiated_Price"]].copy()
    ex["Exception"] = "Price > 115% of Negotiated"
    exceptions.append(ex)

if "On_Time_Delivery" in df.columns:
    ex = df[df["On_Time_Delivery"].str.lower() == "no"].copy()
    ex["Exception"] = "Late Delivery"
    exceptions.append(ex)

if {"Defective_Units", "Quantity"}.issubset(df.columns):
    ex = df[(df["Defective_Units"] / df["Quantity"]) > 0.05].copy()
    ex["Exception"] = "High Defect Rate"
    exceptions.append(ex)

if exceptions:
    exceptions_df = pd.concat(exceptions)

    ex_counts = (
        exceptions_df["Exception"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Exception", "Exception": "Count"})
    )

    fig_ex = px.pie(
        ex_counts,
        names="Exception",
        values="Count",
        title="Exception Distribution"
    )

    c3, c4 = st.columns([1, 2])
    c3.plotly_chart(fig_ex, use_container_width=True)
    c4.dataframe(exceptions_df.head(200))

else:
    st.success("✅ No exceptions detected")

# -----------------------------------------
# Data Preview
# -----------------------------------------
with st.expander("🔍 Preview Filtered Data"):
    st.dataframe(df.head(200))

