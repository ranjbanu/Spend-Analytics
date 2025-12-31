#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SpendAnalyticsDashboard.py
# -----------------------------------------
# Modern Streamlit Spend Analytics Template
# -----------------------------------------
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Spend Analytics + P2P Dashboard",
    layout="wide"
)

DEFAULT_CSV = "Procurement_KPI_Analysis_with_Invoices_projected.csv"

DATE_COLS = [
    "Order_Date", "Delivery_Date", "Price_Effective_Date",
    "Contract_Start_Date", "Contract_End_Date",
    "Invoice_Date", "Invoice_Receipt_Date",
    "Invoice_Due_Date", "Payment_Date"
]

NUM_COLS = [
    "Quantity", "Unit_Price", "Negotiated_Price",
    "Defective_Units", "Tax_Amount", "Freight_Cost",
    "Invoice_Amount", "Projected_Price"
]

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def kpi(container, label, value):
    container.markdown(
        f"""
        <div style="padding:10px;border-radius:8px;
                    background-color:#2e2e2e;
                    color: white; 
                    text-align:center">
            <h4>{label}</h4>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def safe_sum(df, col):
    return df[col].sum() if col in df.columns else 0

    
def parse_dates(df):
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def parse_numbers(df):
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[^\d\-.]", "", regex=True)
                .replace("", np.nan)
                .astype(float)
            )
    return df


def safe_unique(df, col):
    return sorted(df[col].dropna().unique()) if col in df.columns else []

def to_cr(value):
    """Convert to Crores"""
    return value / 1e7

# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data(file):
    if file:
        df = pd.read_csv(file)
    else:
        if not os.path.exists(DEFAULT_CSV):
            st.error("Default CSV not found. Please upload a file.")
            return pd.DataFrame()
        df = pd.read_csv(DEFAULT_CSV)

    df = parse_dates(df)
    df = parse_numbers(df)

    # Derived metrics
    if {"Quantity", "Unit_Price"}.issubset(df.columns):
        df["actual_spend"] = (
            df["Quantity"].fillna(0) * df["Unit_Price"].fillna(0)
            + df.get("Tax_Amount", 0).fillna(0)
            + df.get("Freight_Cost", 0).fillna(0)
        )

    if {"Quantity", "Negotiated_Price"}.issubset(df.columns):
        df["negotiated_spend"] = (
            df["Quantity"].fillna(0) * df["Negotiated_Price"].fillna(0)
        )

    if {"Unit_Price", "Negotiated_Price", "Quantity"}.issubset(df.columns):
        df["savings"] = (
            np.maximum(df["Unit_Price"] - df["Negotiated_Price"], 0)
            * df["Quantity"].fillna(0)
        )

    # Aging bucket
    if "Invoice_Date" in df.columns:
        today = pd.to_datetime(date.today())
        df["Age_Days"] = (today - df["Invoice_Date"]).dt.days

        def bucket(x):
            if pd.isna(x):
                return "Unpaid"
            if x <= 30:
                return "0-30"
            if x <= 60:
                return "31-60"
            if x <= 90:
                return "61-90"
            return ">90"

        df["Aging_Bucket"] = df["Age_Days"].apply(bucket)

    return df

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("📂 Data & Filters")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (optional)", type="csv"
)

df = load_data(uploaded_file)
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
kpi(k1, "Total POs", f"{len(df):,}")

kpi(k2,
    "Actual Spend (₹ Cr)",
    f"{to_cr(safe_sum(df,'actual_spend')):,.2f}"
)

kpi(k3,
    "Savings (₹ Cr)",
    f"{to_cr(safe_sum(df,'savings')):,.2f}"
)

late_pct = (
    (df["On_Time_Delivery"].str.lower() == "no").mean() * 100
    if "On_Time_Delivery" in df.columns else 0
)

kpi(k4,
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
        exceptions_df
        .groupby("Exception")
        .size()
        .reset_index(name="Count")
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

