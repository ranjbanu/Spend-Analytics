
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# ---------------------------
# Page & Theme
# ---------------------------
st.set_page_config(page_title="Spend Analytics & P2P", page_icon="💸", layout="wide")
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
    PPV_Value = ((d["Unit_Price"] - d["Negotiated_Price"]) * d["Quantity"]).sum()
    PPV_Base = (d["Negotiated_Price"] * d["Quantity"]).sum()
    kpis["PPV_Value"] = float(PPV_Value)
    kpis["PPV_Pct"] = float(PPV_Value/PPV_Base*100.0) if PPV_Base>0 else None
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
st.caption(f"Executive overview of procurement spend, savings & risks: {min_inv_date.date() if pd.notnull(min_inv_date) else '—'} → {max_inv_date.date() if pd.notnull(max_inv_date) else '—'}")

# ---------------------------
# KPI row
# ---------------------------
c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Total Spend",f"{fmt_inr(k["total_spend"]):,.2f}")
kpi(c2, "Maverick Spend %",f"{pct(k["maverick_pct"]) if k["maverick_pct"] is not None else "—":,.2f}")
kpi(c3, "On-time Delivery",f"{pct(k["otd_pct"]) if k["otd_pct"] is not None else "—":,.2f}")
kpi(c4, "PPV (vs negotiated)",f"{pct(k["PPV_Pct"]) if k["PPV_Pct"] is not None else "—":,.2f}")
kpi(c5, "Late Payments (₹)",f"{fmt_inr(k["late_spend"]) + fmt_inr(k['late_count']):,.2f}")
#c1.metric("Total Spend", fmt_inr(k["total_spend"]))
#c2.metric("Maverick Spend %", pct(k["maverick_pct"]) if k["maverick_pct"] is not None else "—",help="Off-contract/off-approved spend ÷ addressable spend.")
#c3.metric("On-time Delivery", pct(k["otd_pct"]) if k["otd_pct"] is not None else "—",help="Percent of orders with On_Time_Delivery=='Yes'.")
#c4.metric("PPV (vs negotiated)", pct(k["PPV_Pct"]) if k["PPV_Pct"] is not None else "—",help="Σ(Unit−Negotiated)×Qty ÷ Σ(Negotiated×Qty).")
#c5.metric("Late Payments (₹)", fmt_inr(k["late_spend"]) + f" ({k['late_count']})",help="Payments after due date.")

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

# KPIs within drill slice
dk = calc_kpis(filtered)
d1, d2, d3 = st.columns(3)
d1.metric("Spend (slice)", fmt_inr(dk["total_spend"]))
d2.metric("OTD (slice)", pct(dk["otd_pct"]) if dk["otd_pct"] is not None else "—")
d3.metric("PPV (slice)", pct(dk["PPV_Pct"]) if dk["PPV_Pct"] is not None else "—")

st.dataframe(filtered[[
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
        return (pd.DataFrame(columns=["Item_Category", "PPV_Value", "PPV_Pct", "spend"]),
                pd.DataFrame(columns=["Supplier", "PPV_Value", "PPV_Pct", "spend"]))

    # Row-level terms
    d = d.copy()
    d["PPV_Value"] = (d["Unit_Price"] - d["Negotiated_Price"]) * d["Quantity"]
    d["PPV_Base"] = (d["Negotiated_Price"] * d["Quantity"])

    # By Category (only the current filtered slice)
    cat = d.groupby("Item_Category").agg(
        PPV_Value=("PPV_Value", "sum"),
        base=("PPV_Base", "sum"),
        spend=("Invoice_Amount", "sum")
    ).reset_index()
    cat["PPV_Pct"] = np.where(cat["base"] > 0, cat["PPV_Value"] / cat["base"] * 100.0, np.nan)
    cat = cat.sort_values("PPV_Value", ascending=False)

    # By Supplier (only the current filtered slice)
    sup = d.groupby("Supplier").agg(
        PPV_Value=("PPV_Value", "sum"),
        base=("PPV_Base", "sum"),
        spend=("Invoice_Amount", "sum")
    ).reset_index()
    sup["PPV_Pct"] = np.where(sup["base"] > 0, sup["PPV_Value"] / sup["base"] * 100.0, np.nan)
    sup = sup.sort_values("PPV_Value", ascending=False)

    return cat, sup


st.subheader("Purchase Price Variance (PPV) – by Category & Supplier")

# ---------------------------
# Pie charts for PPV (Selected Period)
# ---------------------------
import plotly.express as px

ppv_cat_sel, ppv_sup_sel = ppv_by_category_and_supplier(filtered)
# Prepare Category pie data (exclude zero/NaN PPV to avoid clutter)
cat_pie = ppv_cat_sel.copy()
cat_pie = cat_pie[(cat_pie["PPV_Value"].notna()) & (cat_pie["PPV_Value"] != 0)]
cat_pie = cat_pie.sort_values("PPV_Value", ascending=False)

# Prepare Supplier pie data (Top-N to keep the pie readable)
TOP_N_SUPPLIERS = st.slider("Top-N suppliers for PPV pie", min_value=1, max_value=5, value=3, step=1)
sup_pie = ppv_sup_sel.copy()
sup_pie = sup_pie[(sup_pie["PPV_Value"].notna()) & (sup_pie["PPV_Value"] != 0)]
sup_pie = sup_pie.sort_values("PPV_Value", ascending=False).head(TOP_N_SUPPLIERS)

# Layout columns
pie_left, pie_right = st.columns(2)

# Pie 1: Category share of PPV value
with pie_left:
    st.caption("PPV value share by Item Category")
    if not cat_pie.empty:
        fig_cat = px.pie(
            cat_pie,
            names="Item_Category",
            values="PPV_Value",
            hole=0.35,
            title="PPV by Category",
        )
        # Rich tooltips
        fig_cat.update_traces(
            hovertemplate="<b>%{label}</b><br>PPV ₹%{value:,.0f}<br>PPV% %{customdata:.2f}%",
            customdata=cat_pie["PPV_Pct"]
        )
        fig_cat.update_layout(legend_title_text="Category")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No PPV data available for the selected period & filters.")

# Pie 2: Supplier share of PPV value (Top-N)
with pie_right:
    st.caption(f"PPV value share by Supplier (Top {TOP_N_SUPPLIERS})")
    if not sup_pie.empty:
        fig_sup = px.pie(
            sup_pie,
            names="Supplier",
            values="PPV_Value",
            hole=0.35,
            title=f"PPV by Supplier (Top {TOP_N_SUPPLIERS})",
        )
        fig_sup.update_traces(
            hovertemplate="<b>%{label}</b><br>PPV ₹%{value:,.0f}<br>PPV% %{customdata:.2f}%",
            customdata=sup_pie["PPV_Pct"]
        )
        fig_sup.update_layout(legend_title_text="Supplier")
        st.plotly_chart(fig_sup, use_container_width=True)
    else:
        st.info("No PPV data available for the selected period & filters.")


c_left, c_right = st.columns(2)

with c_left:
    st.caption("PPV by Category")
    st.dataframe(
        ppv_cat_sel[["Item_Category", "PPV_Value", "PPV_Pct", "spend"]]
        .round({"PPV_Value": 2, "PPV_Pct": 2, "spend": 2}),
        use_container_width=True
    )
    st.download_button(
        "Download CSV (PPV by Category)",
        data=ppv_cat_sel.to_csv(index=False),
        file_name="ppv_by_category_selected_period.csv",
        mime="text/csv"
    )

with c_right:
    st.caption("PPV by Supplier (filtered period)")
    st.dataframe(
        ppv_sup_sel[["Supplier", "PPV_Value", "PPV_Pct", "spend"]]
        .round({"PPV_Value": 2, "PPV_Pct": 2, "spend": 2}),
        use_container_width=True
    )
    st.download_button(
        "Download CSV (PPV by Supplier)",
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
