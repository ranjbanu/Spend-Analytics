
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# ---------------------------
# Page & Theme
# ---------------------------
st.set_page_config(page_title="Spend Analytics & P2P", page_icon="💸", layout="wide")
st.caption("Executive overview of procurement spend, savings & risks")

# ---------------------------
# Utils: currency formatting
# ---------------------------
def fmt_inr(x):
    try:
        return f"₹{x:,.2f}"
    except Exception:
        return "₹0"

def pct(x):
    try:
        return f"{x:.2f}%"
    except Exception:
        return "–"

# ---------------------------
# Data loading & caching
# ---------------------------
@st.cache_data(ttl=900)
def load_data():
    df = pd.read_csv("Procurement_KPI_Analysis_with_Invoices_projected.csv")
    df.columns = [c.strip() for c in df.columns]
    # Parse dates (day-first)
    for col in ['Order_Date','Delivery_Date','Price_Effective_Date','Contract_Start_Date',
                'Contract_End_Date','Invoice_Date','Invoice_Receipt_Date','Invoice_Due_Date','Payment_Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    # Numerics
    num_cols = ['Quantity','Unit_Price','Negotiated_Price','Defective_Units','Tax_Rate','Tax_Amount',
                'Freight_Cost','Lead_Time_Days','Replacement_Return_Cost','Invoice_Amount','Projected_Price']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Helper columns
    df["month"] = df["Invoice_Date"].dt.to_period("M").astype(str)
    df["fy_qtr"] = df["Fiscal_Period"].fillna("")
    df["is_cancelled"] = df["Invoice_Status"].eq("Cancelled")
    df["is_maverick"] = df["Maverick_Flag"].eq("Yes")
    df["is_addressable"] = df["Approved_Category_Flag"].eq("Yes")
    df["is_ontime"] = df["On_Time_Delivery"].eq("Yes")
    df["is_late"] = (df["Payment_Date"].notna() & df["Invoice_Due_Date"].notna() &
                     (df["Payment_Date"] > df["Invoice_Due_Date"]))
    return df

df = load_data()
max_inv_date = pd.to_datetime(df["Invoice_Date"]).max()
min_inv_date = pd.to_datetime(df["Invoice_Date"]).min()

# ---------------------------
# Sidebar (global filters)
# ---------------------------
with st.sidebar:
    st.header("Filters")
    # Period selector
    default_start = (max_inv_date - pd.DateOffset(months=12)).date() if pd.notnull(max_inv_date) else date.today() - timedelta(days=365)
    period = st.date_input("Period", value=(default_start, date.today()))
    cats = st.multiselect("Item Category", options=sorted(df["Item_Category"].dropna().unique().tolist()))
    sups = st.multiselect("Supplier", options=sorted(df["Supplier"].dropna().unique().tolist()))
    inc_cancelled = st.checkbox("Include cancelled invoices?", value=False)
    only_addressable = st.checkbox("Only addressable (Approved_Category_Flag=='Yes')", value=False)
    only_maverick = st.checkbox("Only maverick", value=False)
    apply = st.button("Apply filters")

# ---------------------------
# Apply filters
# ---------------------------
def apply_filters(df):
    start, end = period if isinstance(period, tuple) else (default_start, date.today())
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["Invoice_Date"].dt.date >= start
    if end:
        mask &= df["Invoice_Date"].dt.date <= end
    if cats:
        mask &= df["Item_Category"].isin(cats)
    if sups:
        mask &= df["Supplier"].isin(sups)
    if not inc_cancelled:
        mask &= ~df["is_cancelled"]
    if only_addressable:
        mask &= df["is_addressable"]
    if only_maverick:
        mask &= df["is_maverick"]
    return df[mask].copy()

if not apply:
    st.info("Adjust filters, then click **Apply filters** to refresh KPIs and charts.")
filtered = apply_filters(df) if apply else apply_filters(df)

# ---------------------------
# KPI calculations
# ---------------------------
def calc_kpis(d):
    kpis = {}
    # Spend
    kpis["total_spend"] = float(d["Invoice_Amount"].sum())
    # Maverick (relative to addressable spend in filtered slice)
    addr_spend = float(d.loc[d["is_addressable"], "Invoice_Amount"].sum())
    mav_spend  = float(d.loc[d["is_maverick"], "Invoice_Amount"].sum())
    kpis["addressable_spend"] = addr_spend
    kpis["maverick_spend"] = mav_spend
    kpis["maverick_pct"] = (mav_spend/addr_spend*100.0) if addr_spend>0 else None
    # OTD
    kpis["otd_pct"] = float(d["is_ontime"].mean()*100.0) if len(d)>0 else None
    # PPV
    ppv_val = ((d["Unit_Price"] - d["Negotiated_Price"]) * d["Quantity"]).sum()
    ppv_base = (d["Negotiated_Price"] * d["Quantity"]).sum()
    kpis["ppv_value"] = float(ppv_val)
    kpis["ppv_pct"] = float(ppv_val/ppv_base*100.0) if ppv_base>0 else None
    # Late payments
    late = d[d["is_late"]]
    kpis["late_count"] = int(len(late))
    kpis["late_spend"] = float(late["Invoice_Amount"].sum())
    # Discrepancies
    disc = d["Invoice_Discrepancy_Reason"].fillna("").value_counts()
    kpis["disc_price"] = int(disc.get("Price mismatch", 0))
    kpis["disc_tax"]   = int(disc.get("Tax error", 0))
    kpis["disc_qty"]   = int(disc.get("Quantity mismatch", 0))
    return kpis

k = calc_kpis(filtered)

# ---------------------------
# Header
# ---------------------------
st.title("💸 Spend Analytics Dashboard")
st.caption(f"Data coverage: {min_inv_date.date() if pd.notnull(min_inv_date) else '—'} → {max_inv_date.date() if pd.notnull(max_inv_date) else '—'}")

# ---------------------------
# KPI row
# ---------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Spend", fmt_inr(k["total_spend"]))
c2.metric("Maverick Spend %", pct(k["maverick_pct"]) if k["maverick_pct"] is not None else "—",
          help="Off-contract/off-approved spend ÷ addressable spend.")
c3.metric("On-time Delivery", pct(k["otd_pct"]) if k["otd_pct"] is not None else "—",
          help="Percent of orders with On_Time_Delivery=='Yes'.")
c4.metric("PPV (vs negotiated)", pct(k["ppv_pct"]) if k["ppv_pct"] is not None else "—",
          help="Σ(Unit−Negotiated)×Qty ÷ Σ(Negotiated×Qty).")
c5.metric("Late Payments (₹)", fmt_inr(k["late_spend"]) + f" ({k['late_count']})",
          help="Payments after due date.")

# ---------------------------
# Category & Supplier Pareto
# ---------------------------
st.subheader("Pareto Views")
left, right = st.columns(2)

def pareto(series, topn=15):
    s = series.groupby(series.index).sum() if isinstance(series.index, pd.Index) else series
    s = series
    s = s.sort_values(ascending=False).head(topn)
    return s

with left:
    st.caption("Top Categories by Spend")
    by_cat = (filtered.groupby("Item_Category")["Invoice_Amount"]
              .sum().sort_values(ascending=False).head(15))
    st.bar_chart(by_cat, use_container_width=True)

with right:
    st.caption("Top Suppliers by Spend")
    by_sup = (filtered.groupby("Supplier")["Invoice_Amount"]
              .sum().sort_values(ascending=False).head(15))
    st.bar_chart(by_sup, use_container_width=True)

# ---------------------------
# Trend: monthly spend + maverick share
# ---------------------------
st.subheader("📈 Monthly Trend")
mt = (filtered.groupby(["month"])
      .agg(total_spend=("Invoice_Amount","sum"),
           mav_spend=("is_maverick", lambda s: float(filtered.loc[s.index, "Invoice_Amount"][s].sum())))
      .reset_index())
mt["mav_pct"] = np.where(mt["total_spend"]>0, mt["mav_spend"]/mt["total_spend"]*100.0, np.nan)

t1, t2 = st.columns(2)
with t1:
    st.line_chart(mt.set_index("month")["total_spend"], use_container_width=True)
with t2:
    st.line_chart(mt.set_index("month")["mav_pct"], use_container_width=True)

# ---------------------------
# Drill-down: select category → supplier → documents
# ---------------------------
st.subheader("Drill-down Explorer")
dd_cat = st.selectbox("Category", ["(All)"] + sorted(filtered["Item_Category"].dropna().unique().tolist()))
dd_sup_options = filtered["Supplier"].dropna().unique().tolist()
if dd_cat and dd_cat != "(All)":
    dd_sup_options = filtered.loc[filtered["Item_Category"]==dd_cat, "Supplier"].dropna().unique().tolist()
dd_sup = st.selectbox("Supplier", ["(All)"] + sorted(dd_sup_options))

drill_mask = pd.Series(True, index=filtered.index)
if dd_cat and dd_cat != "(All)":
    drill_mask &= filtered["Item_Category"].eq(dd_cat)
if dd_sup and dd_sup != "(All)":
    drill_mask &= filtered["Supplier"].eq(dd_sup)
drill = filtered[drill_mask].copy()

# KPIs within drill slice
dk = calc_kpis(drill)
d1, d2, d3 = st.columns(3)
d1.metric("Spend (slice)", fmt_inr(dk["total_spend"]))
d2.metric("OTD (slice)", pct(dk["otd_pct"]) if dk["otd_pct"] is not None else "—")
d3.metric("PPV (slice)", pct(dk["ppv_pct"]) if dk["ppv_pct"] is not None else "—")

st.dataframe(drill[[
    "PO_ID","Supplier","Invoice_Number","Invoice_Date","Invoice_Status","Item_Category","Spend_Category",
    "Quantity","Unit_Price","Negotiated_Price","Invoice_Amount","On_Time_Delivery","Maverick_Flag",
    "Invoice_Discrepancy_Reason","Payment_Date","Invoice_Due_Date"
]].sort_values("Invoice_Date", ascending=False), use_container_width=True)

# ---------------------------
# Anomalies Panel
# ---------------------------
st.subheader("🚨 Anomalies & Exceptions")
an_tab1, an_tab2, an_tab3, an_tab4 = st.tabs(["Price mismatch", "Tax error", "Quantity mismatch", "Late payments"])

with an_tab1:
    an1 = filtered[filtered["Invoice_Discrepancy_Reason"].eq("Price mismatch")]
    st.metric("Count", len(an1))
    st.dataframe(an1[[
    "PO_ID","Supplier","Invoice_Number","Invoice_Date","Invoice_Status","Item_Category","Spend_Category",
    "Quantity","Unit_Price","Negotiated_Price","Invoice_Amount","On_Time_Delivery","Maverick_Flag",
    "Invoice_Discrepancy_Reason","Payment_Date","Invoice_Due_Date"
]].sort_values("Invoice_Date", ascending=False), use_container_width=True)
    st.download_button("Download CSV (Price mismatch)", data=an1.to_csv(index=False), file_name="price_mismatch.csv", mime="text/csv")

with an_tab2:
    an2 = filtered[filtered["Invoice_Discrepancy_Reason"].eq("Tax error")]
    st.metric("Count", len(an2))
    st.dataframe(an2[[
    "PO_ID","Supplier","Invoice_Number","Invoice_Date","Invoice_Status","Item_Category","Spend_Category",
    "Quantity","Unit_Price","Negotiated_Price","Invoice_Amount","On_Time_Delivery","Maverick_Flag",
    "Invoice_Discrepancy_Reason","Payment_Date","Invoice_Due_Date"
]].sort_values("Invoice_Date", ascending=False), use_container_width=True)
    st.download_button("Download CSV (Tax error)", data=an2.to_csv(index=False), file_name="tax_error.csv", mime="text/csv")

with an_tab3:
    an3 = filtered[filtered["Invoice_Discrepancy_Reason"].eq("Quantity mismatch")]
    st.metric("Count", len(an3))
    st.dataframe(an3[[
    "PO_ID","Supplier","Invoice_Number","Invoice_Date","Invoice_Status","Item_Category","Spend_Category",
    "Quantity","Unit_Price","Negotiated_Price","Invoice_Amount","On_Time_Delivery","Maverick_Flag",
    "Invoice_Discrepancy_Reason","Payment_Date","Invoice_Due_Date"
]].sort_values("Invoice_Date", ascending=False), use_container_width=True)
    st.download_button("Download CSV (Quantity mismatch)", data=an3.to_csv(index=False), file_name="qty_mismatch.csv", mime="text/csv")

with an_tab4:
    late = filtered[filtered["is_late"]]
    st.metric("Late payments", len(late))
    st.metric("Late payment spend", fmt_inr(late["Invoice_Amount"].sum()))
    st.dataframe(late[["Supplier","Invoice_Number","Invoice_Date","Invoice_Due_Date","Payment_Date","Invoice_Amount"]]
                 .sort_values("Payment_Date", ascending=False), use_container_width=True)
    st.download_button("Download CSV (Late payments)", data=late.to_csv(index=False), file_name="late_payments.csv", mime="text/csv")

# ---------------------------
# PPV by category / supplier
# ---------------------------

# ---------------------------
# PPV for the selected period (respects sidebar filters)
# ---------------------------
def ppv_by_category_and_supplier(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes PPV over the FILTERED slice `d`.
      PPV value = Σ (Unit_Price - Negotiated_Price) * Quantity
      PPV %     = PPV value ÷ Σ (Negotiated_Price * Quantity)

    Returns (ppv_cat_df, ppv_sup_df) both sorted by PPV value desc.
    """
    if d.empty:
        return (pd.DataFrame(columns=["Item_Category", "ppv_value", "ppv_pct", "spend"]),
                pd.DataFrame(columns=["Supplier", "ppv_value", "ppv_pct", "spend"]))

    # Row-level terms
    d = d.copy()
    d["ppv_val"] = (d["Unit_Price"] - d["Negotiated_Price"]) * d["Quantity"]
    d["ppv_base"] = (d["Negotiated_Price"] * d["Quantity"])

    # By Category (only the current filtered slice)
    cat = d.groupby("Item_Category").agg(
        ppv_value=("ppv_val", "sum"),
        base=("ppv_base", "sum"),
        spend=("Invoice_Amount", "sum")
    ).reset_index()
    cat["ppv_pct"] = np.where(cat["base"] > 0, cat["ppv_value"] / cat["base"] * 100.0, np.nan)
    cat = cat.sort_values("ppv_value", ascending=False)

    # By Supplier (only the current filtered slice)
    sup = d.groupby("Supplier").agg(
        ppv_value=("ppv_val", "sum"),
        base=("ppv_base", "sum"),
        spend=("Invoice_Amount", "sum")
    ).reset_index()
    sup["ppv_pct"] = np.where(sup["base"] > 0, sup["ppv_value"] / sup["base"] * 100.0, np.nan)
    sup = sup.sort_values("ppv_value", ascending=False)

    return cat, sup


st.subheader("PPV (Selected Period) – by Category & Supplier")

ppv_cat_sel, ppv_sup_sel = ppv_by_category_and_supplier(filtered)

c_left, c_right = st.columns(2)

with c_left:
    st.caption("PPV by Category (filtered period)")
    st.dataframe(
        ppv_cat_sel[["Item_Category", "ppv_value", "ppv_pct", "spend"]]
        .round({"ppv_value": 2, "ppv_pct": 2, "spend": 2}),
        use_container_width=True
    )
    st.download_button(
        "Download CSV (PPV by Category – selected period)",
        data=ppv_cat_sel.to_csv(index=False),
        file_name="ppv_by_category_selected_period.csv",
        mime="text/csv"
    )

with c_right:
    st.caption("PPV by Supplier (filtered period)")
    st.dataframe(
        ppv_sup_sel[["Supplier", "ppv_value", "ppv_pct", "spend"]]
        .round({"ppv_value": 2, "ppv_pct": 2, "spend": 2}),
        use_container_width=True
    )
    st.download_button(
        "Download CSV (PPV by Supplier – selected period)",
        data=ppv_sup_sel.to_csv(index=False),
        file_name="ppv_by_supplier_selected_period.csv",
        mime="text/csv"
    )

# ---------------------------
# Definitions & Data freshness
# ---------------------------
with st.expander("Definitions & Data Freshness"):
    st.markdown(f"""
**Data last loaded:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Total Spend:** Sum of `Invoice_Amount` in the filtered slice.
- **Maverick Spend %:** Sum of `Invoice_Amount` where `Maverick_Flag=='Yes'` ÷ sum of `Invoice_Amount` where `Approved_Category_Flag=='Yes'`.
- **OTD %:** Share of rows with `On_Time_Delivery=='Yes'`.
- **PPV %:** Σ(Unit−Negotiated)×Qty ÷ Σ(Negotiated×Qty) over filtered rows.
- **Late Payments:** `Payment_Date > Invoice_Due_Date`.
    """)
