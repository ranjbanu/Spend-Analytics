#!/usr/bin/env python
# coding: utf-8

# In[8]:



# app.py
# Spend Analytics + P2P Dashboard (Streamlit + Pandas)

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

st.title("📊 Spend Analytics & P2P Dashboard")

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


filters = {
    "Item_Category": st.sidebar.multiselect("Item Category", safe_unique(df, "Item_Category")),
    "Supplier": st.sidebar.multiselect("Supplier", safe_unique(df, "Supplier")),
    "Supplier_Region": st.sidebar.multiselect("Supplier Region", safe_unique(df, "Supplier_Region")),
    "Spend_Category": st.sidebar.multiselect("Spend Category", safe_unique(df, "Spend_Category")),
    "Spend_Class": st.sidebar.multiselect("Spend Class", safe_unique(df, "Spend_Class")),
    "Fiscal_Period": st.sidebar.multiselect("Fiscal Period", safe_unique(df, "Fiscal_Period")),
    "Invoice_Status": st.sidebar.multiselect("Invoice Status", safe_unique(df, "Invoice_Status")),
    "Aging_Bucket": st.sidebar.multiselect("Aging Bucket", safe_unique(df, "Aging_Bucket")),
}

mask = pd.Series(True, index=df.index)
for col, vals in filters.items():
    if vals and col in df.columns:
        mask &= df[col].isin(vals)

dff = df[mask].copy()

# --------------------------------------------------
# KPI Cards
# --------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.beta_columns(6)

kpi(c1, "Total POs", f"{len(dff):,}")

if "Order_Status" in dff.columns:
    delivered_pct = dff["Order_Status"].eq("Delivered").mean() * 100
    kpi(c2, "Delivered %", f"{delivered_pct:.1f}%")
else:
    kpi(c2, "Delivered %", "N/A")

if "Compliance" in dff.columns:
    compliance_pct = dff["Compliance"].str.lower().eq("yes").mean() * 100
    kpi(c3, "Compliance %", f"{compliance_pct:.1f}%")
else:
    kpi(c3, "Compliance %", "N/A")

if "On_Time_Delivery" in dff.columns:
    otd_pct = dff["On_Time_Delivery"].str.lower().eq("yes").mean() * 100
    kpi(c4, "OTD %", f"{otd_pct:.1f}%")
else:
    kpi(c4, "OTD %", "N/A")

kpi(c5, "Actual Spend (₹)", f"{safe_sum(dff, 'actual_spend')/100000:,.2f}")
kpi(c6, "Savings (₹)", f"{safe_sum(dff, 'savings')/100000:,.2f}")



# --------------------------------------------------
# Charts
# --------------------------------------------------
col1, col2 = st.beta_columns(2)

if "Item_Category" in dff.columns:
    fig = px.bar(
        dff.groupby("Item_Category")["actual_spend"].sum().reset_index(),
        x="Item_Category",
        y="actual_spend",
        title="Spend by Item Category"
    )
    col1.plotly_chart(fig, use_container_width=True)

if "Supplier" in dff.columns:
    fig = px.bar(
        dff.groupby("Supplier")["actual_spend"]
        .sum().sort_values(ascending=False).head(10).reset_index(),
        x="Supplier",
        y="actual_spend",
        title="Top 10 Suppliers by Spend"
    )
    col2.plotly_chart(fig, use_container_width=True)

col1, col2 = st.beta_columns(2)
# --------------------------------------------------
# Aging
# --------------------------------------------------
if {"Aging_Bucket", "Invoice_Amount"}.issubset(dff.columns):
    fig = px.pie(
        dff.groupby("Aging_Bucket")["Invoice_Amount"].sum().reset_index(),
        names="Aging_Bucket",
        values="Invoice_Amount",
        title="Invoice Aging (in days)"
    )
    col1.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Exceptions
# --------------------------------------------------

exceptions = []

if {"Unit_Price", "Negotiated_Price"}.issubset(dff.columns):
    ex = dff[dff["Unit_Price"] > 1.15 * dff["Negotiated_Price"]]
    if not ex.empty:
        ex["Exception"] = "Price > 115% of negotiated"
        exceptions.append(ex)

if "On_Time_Delivery" in dff.columns:
    ex = dff[dff["On_Time_Delivery"].str.lower() == "no"]
    if not ex.empty:
        ex["Exception"] = "Late delivery"
        exceptions.append(ex)

if {"Defective_Units", "Quantity"}.issubset(dff.columns):
    ex = dff[(dff["Defective_Units"] / dff["Quantity"]) > 0.05]
    if not ex.empty:
        ex["Exception"] = "High defect rate"
        exceptions.append(ex)

# --------------------------------------------------
# Exceptions
# --------------------------------------------------
if exceptions:
    exceptions_df = pd.DataFrame(pd.concat(exceptions))

    # 1️⃣ Count of each exception type
    exception_counts = (
        exceptions_df.groupby("Exception").size()
        .reset_index(name="Count")
    )
    if exception_counts["Count"].sum() > 0:
        fig = px.pie(
            exception_counts,
            names="Exception",
            values="Count",
            title="Exceptions Breakdown",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        col2.plotly_chart(fig)
        
st.subheader("⚠️ Exceptions")
        
if exceptions:
    st.dataframe(pd.concat(exceptions))
else:
    st.info("No exceptions found.")

# --------------------------------------------------
# Download
# --------------------------------------------------
#st.download_button(
#    "⬇️ Download Filtered Data",
#    "filtered_spend_p2p.csv",
#    "text/csv"
#)


# In[ ]:




