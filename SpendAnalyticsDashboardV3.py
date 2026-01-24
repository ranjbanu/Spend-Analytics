
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

import plotly.graph_objects as go
import os
import re
import json

def _weighted_majority(labels: np.ndarray, weights: np.ndarray):
    # Return label with max total weight (ties broken by overall frequency)
    label_weights = {}
    for lab, w in zip(labels, weights):
        label_weights[lab] = label_weights.get(lab, 0.0) + float(w)
    return max(label_weights.items(), key=lambda kv: kv[1])[0]
    
def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 &/_\-]+", " ", s)  # keep common separators
    s = re.sub(r"\s+", " ", s).strip()
    return s

import numpy as np

# Load once at startup (fast)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str):
    """Generate a local embedding using MiniLM."""
    if not text:
        return np.zeros(384)  # MiniLM dimension
    return embed_model.encode(text, convert_to_numpy=True)



def fit_memory(df: pd.DataFrame):
    d = df.copy()
    d = d[(d["Item_Category"].notna()) & (d["Spend_Category"].notna())].copy()

    sup_mode_item = (
        d.groupby("Supplier")["Item_Category"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    sup_mode_spend = (
        d.groupby("Supplier")["Spend_Category"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    text = _build_text_column(d)

    # --- TF-IDF ---
    vect = TfidfVectorizer(min_df=1, ngram_range=(2, 5))
    X = vect.fit_transform(text.values)

    # --- LLM Embeddings ---
    embeddings = np.vstack([get_embedding(t) for t in text])

    model = {
        "df": d,
        "sup_mode_item": sup_mode_item,
        "sup_mode_spend": sup_mode_spend,
        "vectorizer": vect,
        "X": X,
        "embeddings": embeddings
    }
    return model

def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-10))

def predict(model, product_name: str, supplier: str, top_k: int = 5):

    supplier = supplier or ""
    product_name = product_name or ""

    text_in = _clean_text(f"{supplier} {product_name}")

    # fallbacks
    fallback_item = model["sup_mode_item"].get(supplier) or model["df"]["Item_Category"].mode().iat[0]
    fallback_spend = model["sup_mode_spend"].get(supplier) or model["df"]["Spend_Category"].mode().iat[0]

    if not text_in:
        return {
            "Item_Category": fallback_item,
            "Spend_Category": fallback_spend,
            "method": "fallback-empty-text"
        }

    # --- TF-IDF similarity ---
    x_tfidf = model["vectorizer"].transform([text_in])
    sim_tfidf = (model["X"] @ x_tfidf.T).toarray().ravel()

    # --- LLM embedding similarity ---
    embed_in = get_embedding(text_in)
    sim_embed = np.array([
        cosine_sim(embed_in, emb) for emb in model["embeddings"]
    ])

    # --- Final hybrid similarity ---
    sim_final = 0.5 * sim_tfidf + 0.5 * sim_embed



    sim_final = np.nan_to_num(sim_final, nan=0.0, posinf=0.0, neginf=0.0)
    idx_all = np.argsort(-sim_final)
    idx_top = idx_all[:top_k]
    sims_top = sim_final[idx_top]

    # keep only strictly positive sims (0 means no similarity)
    mask = sims_top > 0
    if not np.any(mask):
        return {
            "Item_Category": fallback_item,
            "Spend_Category": fallback_spend,
            "method": "supplier_mode_fallback_no_sim",
            "support": None,
        }

    idx = idx_top[mask]
    sims = sims_top[mask]

    # Guard: if something still goes wrong, fallback cleanly
    if len(idx) == 0:
        return {
            "Item_Category": fallback_item,
            "Spend_Category": fallback_spend,
            "method": "supplier_mode_fallback_mask_empty",
            "support": None,
        }
    cats = model["df"].iloc[idx]["Item_Category"].values
    spends = model["df"].iloc[idx]["Spend_Category"].values

    pred_item = _weighted_majority(cats, sims)
    pred_spend = _weighted_majority(spends, sims)

    return {
        "Item_Category": pred_item,
        "Spend_Category": pred_spend,
        "method": "hybrid-llm-tfidf",
        "support": {
            "top_examples": model["df"].iloc[idx][
                ["Supplier", "Item_Description", "Item_Category", "Spend_Category"]
            ].assign(similarity=np.round(sims, 3)).head(3).to_dict(orient="records")
        }
    }

def _build_text_column(df: pd.DataFrame) -> pd.Series:
    # Combine Supplier and Item_Description as model text
    sup = df.get("Supplier", "").fillna("").astype(str)
    desc = df.get("Item_Description", "").fillna("").astype(str)
    text = (sup + " | " + desc).map(_clean_text)
    return text


def pareto_figure_from_series(series: pd.Series, title: str, topn: int = 15):
    """
    Build a Pareto chart: bars = value (left axis), line = cumulative % (right axis).
    'series' should be a 1-D series indexed by category/supplier with numeric values.
    """
    s = series.dropna().groupby(level=0).sum() if isinstance(series.index, pd.MultiIndex) else series.dropna()
    s = s.sort_values(ascending=False).head(topn)

    dfp = pd.DataFrame({"label": s.index.astype(str), "value": s.values})
    dfp["cum_value"] = dfp["value"].cumsum()
    total = dfp["value"].sum()
    dfp["cum_pct"] = (dfp["cum_value"] / total) * 100.0 if total != 0 else 0.0

    fig = go.Figure()

    # Bars (left axis)
    fig.add_trace(
        go.Bar(
            x=dfp["label"],
            y=dfp["value"],
            marker_color="#1f77b4",
            yaxis="y1",
            hovertemplate="<b>%{x}</b><br>Value: %{y:,.0f}<extra></extra>",
        )
    )

    # Cumulative % (right axis)
    fig.add_trace(
        go.Scatter(
            x=dfp["label"],
            y=dfp["cum_pct"],
            mode="lines+markers",
            line=dict(color="#d62728", width=2),
            marker=dict(size=7, color="#d62728"),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Cum %: %{y:.1f}%<extra></extra>",
        )
    )

    # Layout: dual axes + 80% rule (optional)
    fig.update_layout(
        title=title,
        bargap=0.2,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(title="", tickangle=-45, showgrid=False),
        yaxis=dict(title="Value", rangemode="tozero", showgrid=True),
        yaxis2=dict(overlaying="y", side="right", range=[0, 100], showgrid=False),
        height=480,
    )

    # Add horizontal 80% reference line
    fig.add_hline(
        y=80, line_dash="dash", line_color="gray",
        annotation_text="80%", annotation_position="top left", secondary_y=True
    )

    return fig


# ---------- Supplier Optimization: KPIs + Scoring ----------

import pandas as pd
import numpy as np

def _use_negotiated(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    if "Negotiated_Price" in d.columns:
        d["neg_used"] = d["Negotiated_Price"]
    return d


def _minmax_by_category(series: pd.Series, by_key: pd.Series) -> pd.Series:
    """
    Min-max normalize within each Item_Category group to [0,1].
    Returns a Series aligned to the original index.
    For groups with constant or non-finite ranges, return 0.5 (neutral).
    """
    s = series.astype(float)
    gmin = s.groupby(by_key).transform("min")
    gmax = s.groupby(by_key).transform("max")
    rng = gmax - gmin

    norm = (s - gmin) / rng
    # neutral 0.5 when range is 0 or NaN
    norm = norm.where(rng > 0, 0.5)
    # replace remaining non-finites
    norm = norm.fillna(0.5)
    return norm


def compute_supplier_kpis(d: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates supplier KPIs per Item_Category:
    - PPV_Value, PPV_Base, PPV_Pct
    - OTD rate, discrepancy rate, late-payment rate
    - Spend, Quantity, Txn count
    """
    if d.empty:
        return pd.DataFrame(columns=[
            "Item_Category","Supplier","spend","qty","txn_count",
            "PPV_Value","ppv_base","PPV_Pct","otd_rate","disc_rate","late_rate"
        ])

    d = _use_negotiated(d)

    # Row-level PPV terms
    d["PPV_Value_row"] = (d["Unit_Price"] - d["neg_used"]) * d["Quantity"]
    d["PPV_Base_row"]  = (d["neg_used"] * d["Quantity"])

    # Boolean OTD
    otd_bool = d["On_Time_Delivery"].astype(str).str.lower().eq("yes")
    d["otd_bool"] = otd_bool

    # Aggregation per Item_Category + Supplier
    grp = d.groupby(["Item_Category","Supplier"], dropna=False)
    out = grp.agg(
        spend=("Invoice_Amount","sum"),
        qty=("Quantity","sum"),
        txn_count=("PO_ID","count"),
        PPV_Value=("PPV_Value_row","sum"),
        ppv_base=("PPV_Base_row","sum"),
        otd_rate=("otd_bool","mean"),
        disc_rate=("Invoice_Discrepancy_Reason", lambda s: s.notna().mean()),
        late_rate=("is_late","mean")
    ).reset_index()

    # PPV%
    out["PPV_Pct"] = np.where(out["ppv_base"] > 0, out["PPV_Value"] / out["ppv_base"] * 100.0, np.nan)

    return out

def score_suppliers(kpis_df: pd.DataFrame, weights: dict, cost_mix_PPV_Pct: float = 0.7) -> pd.DataFrame:
    """
    Scores suppliers per category with a composite 0–100 score.
    weights keys: {'cost','reliability','risk','volume'} in [0,1] (we’ll normalize).
    cost_mix_PPV_Pct: share of PPV% vs PPV value in cost sub-score (default 70%).
    """
    dfk = kpis_df.copy()

    # Prepare normalization keys
    cat_key = dfk["Item_Category"]

    # --- Cost sub-score ---
    # Normalize PPV% within category (lower is better → invert)
    PPV_Pct_norm = _minmax_by_category(dfk["PPV_Pct"].fillna(0.0), cat_key)
    cost_PPV_Pct_score = 1.0 - PPV_Pct_norm

    # Penalize only unfavorable PPV value (positive PPV_Value) → normalize
    unf_ppv_val = dfk["PPV_Value"].clip(lower=0)  # negatives (favorable) become 0
    unf_ppv_val_norm = _minmax_by_category(unf_ppv_val.fillna(0.0), cat_key)
    cost_ppv_val_score = 1.0 - unf_ppv_val_norm

    dfk["cost_score"] = cost_mix_PPV_Pct * cost_PPV_Pct_score + (1.0 - cost_mix_PPV_Pct) * cost_ppv_val_score

    # --- Reliability sub-score ---
    rel_norm = _minmax_by_category(dfk["otd_rate"].fillna(0.0), cat_key)
    dfk["reliability_score"] = rel_norm  # higher OTD is better

    # --- Risk sub-score ---
    # Composite risk = 0.5 * discrepancy + 0.5 * late
    risk_raw = 0.5 * dfk["disc_rate"].fillna(0.0) + 0.5 * dfk["late_rate"].fillna(0.0)
    risk_norm = _minmax_by_category(risk_raw, cat_key)
    dfk["risk_score"] = 1.0 - risk_norm  # lower risk better

    # --- Volume sub-score ---
    # Combine spend & qty (equal weight), then normalize within category
    vol_raw = 0.5 * dfk["spend"].fillna(0.0) + 0.5 * dfk["qty"].fillna(0.0)
    vol_norm = _minmax_by_category(vol_raw, cat_key)
    dfk["volume_score"] = vol_norm  # larger volume → better fit/capacity

    # Normalize weights to sum 1
    w_cost = float(weights.get("cost", 0.4))
    w_rel  = float(weights.get("reliability", 0.3))
    w_risk = float(weights.get("risk", 0.2))
    w_vol  = float(weights.get("volume", 0.1))
    w_sum = max(w_cost + w_rel + w_risk + w_vol, 1e-6)
    w_cost, w_rel, w_risk, w_vol = (w_cost/w_sum, w_rel/w_sum, w_risk/w_sum, w_vol/w_sum)

    # Final composite score (0–100)
    dfk["score_0_1"] = (
        w_cost * dfk["cost_score"] +
        w_rel  * dfk["reliability_score"] +
        w_risk * dfk["risk_score"] +
        w_vol  * dfk["volume_score"]
    )
    dfk["Supplier_Score"] = (dfk["score_0_1"] * 100.0).round(2)

    # Reason codes (brief)
    def _reason(row):
        reasons = []
        if row["cost_score"] >= 0.7: reasons.append("Low unfavorable PPV")
        if row["reliability_score"] >= 0.7: reasons.append(f"High OTD ({row['otd_rate']*100:.1f}%)")
        if row["risk_score"] >= 0.7: reasons.append("Low discrepancy/late risk")
        if row["volume_score"] >= 0.7: reasons.append(f"Strong volume fit (₹{row['spend']:,.0f}, qty {row['qty']:.0f})")
        return "; ".join(reasons) if reasons else "Balanced performance"

    dfk["Rationale"] = dfk.apply(_reason, axis=1)

    return dfk

# =========================
# Time-series: SARIMA forecast by category (simplified, no Holt-Winters)
# =========================

def _ensure_month_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a proper monthly Period column named 'month'."""
    d = df.copy()
    if "month" in d.columns:
        if d["month"].dtype == object:
            d["month"] = pd.PeriodIndex(d["month"], freq="M")
    else:
        d["Invoice_Date"] = pd.to_datetime(d["Invoice_Date"], dayfirst=True, errors="coerce")
        d["month"] = d["Invoice_Date"].dt.to_period("M")
    return d

def _reindex_months_continuous(series: pd.Series) -> pd.Series:
    """Ensure monthly continuity; fill missing months with 0 spend."""
    if series.empty:
        return series
    idx = pd.period_range(series.index.min(), series.index.max(), freq="M")
    return series.reindex(idx).fillna(0.0)

def _sarima_best(series: pd.Series, season: int = 12):
    """
    Fit a small SARIMA grid and return the best by AIC.
    Grid: (p,d,q) in {0,1} x {0,1} x {0,1}, (P,D,Q) in {0,1} with seasonal period.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        return None  # statsmodels missing; caller will fallback

    y = series.astype(float)
    d_candidates = [0, 1]
    D_candidates = [0, 1]
    p_candidates = q_candidates = P_candidates = Q_candidates = [0, 1]

    best_model = None
    best_aic = np.inf
    for p in p_candidates:
        for d in d_candidates:
            for q in q_candidates:
                for P in P_candidates:
                    for D in D_candidates:
                        for Q in Q_candidates:
                            order = (p, d, q)
                            sorder = (P, D, Q, season)
                            try:
                                mod = SARIMAX(
                                    y, order=order, seasonal_order=sorder,
                                    enforce_stationarity=False, enforce_invertibility=False
                                )
                                res = mod.fit(disp=False)
                                if res.aic < best_aic and np.isfinite(res.aic):
                                    best_aic = res.aic
                                    best_model = res
                            except Exception:
                                continue
    return best_model  # may be None

def _seasonal_naive(series: pd.Series, horizon: int = 3, season: int = 12):
    """
    Seasonal naive baseline (used only as fallback when SARIMA can't fit).
    ŷ_{t+h} = y_{t+h-season} if available, else last observed.
    CI estimated from residual std vs seasonal lag (indicative only).
    """
    y = series.astype(float)
    n = len(y)
    residuals = []
    if n > season:
        residuals = [y.iloc[i] - y.iloc[i - season] for i in range(season, n)]
    res_std = float(np.nanstd(residuals)) if residuals else float(np.nanstd(y))
    z80, z95 = 1.28, 1.96

    f_vals = []
    for h in range(1, horizon + 1):
        src_idx = n - season + h - 1
        if 0 <= src_idx < n:
            val = float(y.iloc[src_idx])
        else:
            val = float(y.iloc[-1]) if n else 0.0
        f_vals.append(val)

    ci80 = [(v - z80 * res_std, v + z80 * res_std) for v in f_vals]
    ci95 = [(v - z95 * res_std, v + z95 * res_std) for v in f_vals]
    return f_vals, ci80, ci95

def forecast_by_category_timeseries_simple(df: pd.DataFrame,
                                           horizon: int = 3,
                                           season: int = 12,
                                           min_points: int = 6,
                                           exclude_cancelled: bool = True) -> pd.DataFrame:
    """
    Forecast monthly spend by Item_Category for next 'horizon' months using SARIMA.
    If SARIMA can't fit or history is too short, use seasonal-naive fallback.

    Returns (all rounded to 2 decimals):
      Item_Category, Forecast_Month, Forecast, CI80_Low, CI80_High, CI95_Low, CI95_High, Model
    """
    d = _ensure_month_col(df)

    if exclude_cancelled and "Invoice_Status" in d.columns:
        d = d[~d["Invoice_Status"].eq("Cancelled")].copy()

    by_cat = (d.groupby(["Item_Category", "month"])["Invoice_Amount"]
                .sum().reset_index())

    results = []
    for cat, gcat in by_cat.groupby("Item_Category"):
        series = gcat.set_index("month")["Invoice_Amount"].sort_index()
        series = _reindex_months_continuous(series)

        model_name = ""
        f_vals = ci80_low = ci80_high = ci95_low = ci95_high = None

        # Try SARIMA if enough points; else fallback
        sarima_res = None
        if len(series) >= min_points:
            sarima_res = _sarima_best(series, season=season)

        if sarima_res is not None:
            try:
                fc = sarima_res.get_forecast(steps=horizon)
                f_vals = list(map(float, fc.predicted_mean))
                ci95_df = fc.conf_int(alpha=0.05)
                ci80_df = fc.conf_int(alpha=0.20)
                ci95_low = list(map(float, ci95_df.iloc[:, 0]))
                ci95_high = list(map(float, ci95_df.iloc[:, 1]))
                ci80_low = list(map(float, ci80_df.iloc[:, 0]))
                ci80_high = list(map(float, ci80_df.iloc[:, 1]))
                model_name = "SARIMA"
            except Exception:
                f_vals, ci80, ci95 = _seasonal_naive(series, horizon=horizon, season=season)
                ci80_low, ci80_high = [c[0] for c in ci80], [c[1] for c in ci80]
                ci95_low, ci95_high = [c[0] for c in ci95], [c[1] for c in ci95]
                model_name = "Seasonal Naive (fallback)"
        else:
            f_vals, ci80, ci95 = _seasonal_naive(series, horizon=horizon, season=season)
            ci80_low, ci80_high = [c[0] for c in ci80], [c[1] for c in ci80]
            ci95_low, ci95_high = [c[0] for c in ci95], [c[1] for c in ci95]
            model_name = "Seasonal Naive (fallback)"

        last_per = series.index.max() if not series.empty else pd.Period(pd.Timestamp.today(), freq="M")
        future_periods = [last_per + i for i in range(1, horizon + 1)]

        for i in range(horizon):
            results.append({
                "Item_Category": cat,
                "Forecast_Month": future_periods[i].strftime("%Y-%m"),
                "Forecast": round(float(f_vals[i]), 2),
            })

    return pd.DataFrame(results)


# ---------------------------
# Page & Theme
# ---------------------------
tabs = st.tabs(["💸 Dashboard", "📈 Forecasting","🤝 Supplier Optimization","🗂️ Auto Categorize"])
with tabs[0]:
    st.set_page_config(page_title="Spend Analytics & P2P", page_icon="💸", layout="wide")
    # ---------------------------
    # Utils: currency formatting
    # ---------------------------
    def fmt_inr(x):
        try:
            return f"₹{x/1000000:,.2f} mn"
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
    
    # ---------- PPV Enhancements (helpers) ----------
    
    def compute_ppv_terms(d: pd.DataFrame, tol_pct: float = 0.001) -> pd.DataFrame:
        """
        Compute PPV terms with a small tolerance:
        - PPV_Base = Negotiated_Price * Quantity
        - PPV_Value = (Unit_Price - Negotiated_Price) * Quantity, but set to 0
          when price difference is within tol_pct of negotiated (to avoid rounding noise).
        """
        d = d.copy()
        d["PPV_Base"] = d["Negotiated_Price"] * d["Quantity"]
        raw_ppv = (d["Unit_Price"] - d["Negotiated_Price"]) * d["Quantity"]
    
        small_diff = (
            d["Unit_Price"].notna() & d["Negotiated_Price"].notna() &
            (np.abs(d["Unit_Price"] - d["Negotiated_Price"]) <= d["Negotiated_Price"].abs() * tol_pct)
        )
        d["PPV_Value"] = np.where(small_diff, 0.0, raw_ppv)
        return d
    
    def ppv_favorable_unfavorable(d: pd.DataFrame) -> tuple[float, float, float]:
        """
        Return (favorable_sum, unfavorable_sum, net_sum) in currency terms.
        Favorable rows have PPV_Value < 0, unfavorable > 0.
        """
        fav = float(d.loc[d["PPV_Value"] < 0, "PPV_Value"].sum())
        unf = float(d.loc[d["PPV_Value"] > 0, "PPV_Value"].sum())
        net = float(d["PPV_Value"].sum())
        return fav, unf, net
    
    # ---------------------------
    # Data loading & caching
    # ---------------------------
    @st.cache_data(ttl=900)
    def load_data():
        df = pd.read_csv("Procurement_KPI_Analysis_with_Invoices_Projected_NegotiatedPrice.csv")
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
    base_df = df.copy() 
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
    kpi(c1, "Total Spend",f"{fmt_inr(k["total_spend"])}")
    kpi(c2, "Maverick Spend %",f"{pct(k["maverick_pct"]) if k["maverick_pct"] is not None else "—"}")
    kpi(c3, "On-time Delivery",f"{pct(k["otd_pct"]) if k["otd_pct"] is not None else "—"}")
    kpi(c4, "PPV (vs negotiated)",f"{pct(k["PPV_Pct"]) if k["PPV_Pct"] is not None else "—"}")
    kpi(c5, "Late Payments",f"{fmt_inr(k["late_spend"])}")
    
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
        st.caption("Top Categories by Spend (Pareto)")
        by_cat = filtered.groupby("Item_Category")["Invoice_Amount"].sum()
        fig_cat_pareto = pareto_figure_from_series(by_cat, title="Pareto: Spend by Category", topn=10)
        st.plotly_chart(fig_cat_pareto, use_container_width=True)
    
    with right:
        st.caption("Top Suppliers by Spend (Pareto)")
        by_sup = filtered.groupby("Supplier")["Invoice_Amount"].sum()
        fig_sup_pareto = pareto_figure_from_series(by_sup, title="Pareto: Spend by Supplier", topn=10)
        st.plotly_chart(fig_sup_pareto, use_container_width=True)
    
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
    
    # ---------------------------
    # Pie charts for PPV (Selected Period)
    # ---------------------------
    import plotly.express as px
    
    # =============================================================
    # Purchase Price Variance (PPV) – by Category & Supplier (Enhanced)
    # =============================================================
    st.subheader("Purchase Price Variance (PPV) – by Category & Supplier")
    
    data_for_ppv = filtered.copy()
    # Compute PPV terms with tolerance to avoid cosmetic noise
    TOL_PCT = 0.001  # 0.1% tolerance
    data_for_ppv = compute_ppv_terms(data_for_ppv, tol_pct=TOL_PCT)
    
    # Aggregate PPV by Category & Supplier
    ppv_cat = (data_for_ppv.groupby("Item_Category", dropna=False)
               .agg(PPV_Value=("PPV_Value","sum"),
                    base=("PPV_Base","sum"),
                    spend=("Invoice_Amount","sum"))
               .reset_index())
    ppv_cat["PPV_Pct"] = np.where(ppv_cat["base"].abs()>1e-9, ppv_cat["PPV_Value"]/ppv_cat["base"]*100.0, np.nan)
    ppv_cat = ppv_cat.sort_values("PPV_Value", ascending=True)  # show favorable first (more negative)
    
    ppv_sup = (data_for_ppv.groupby("Supplier", dropna=False)
               .agg(PPV_Value=("PPV_Value","sum"),
                    base=("PPV_Base","sum"),
                    spend=("Invoice_Amount","sum"))
               .reset_index())
    ppv_sup["PPV_Pct"] = np.where(ppv_sup["base"].abs()>1e-9, ppv_sup["PPV_Value"]/ppv_sup["base"]*100.0, np.nan)
    ppv_sup = ppv_sup.sort_values("PPV_Value", ascending=True)
    
    # KPI tiles: Favorable vs Unfavorable vs Net
    fav_val, unf_val, net_val = ppv_favorable_unfavorable(data_for_ppv)
    c1, c2, c3 = st.columns(3)
    c1.metric("Favorable PPV (₹)", fmt_inr(abs(fav_val)), help="Total savings vs negotiated (PPV < 0)")
    c2.metric("Unfavorable PPV (₹)", fmt_inr(unf_val), help="Total paid above negotiated (PPV > 0)")
    c3.metric("Net PPV (₹)", fmt_inr(net_val), help="Favorable + Unfavorable")
    
    # --- Visuals: pies & bars ---
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Category pie (absolute values + sign in hover)
    cat_pie = ppv_cat.copy()
    cat_pie["PPV_Abs"] = cat_pie["PPV_Value"].abs()
    cat_pie["Sign"] = np.where(cat_pie["PPV_Value"] < 0, "Favorable", "Unfavorable")
    cat_pie = cat_pie[cat_pie["PPV_Abs"] > 0].sort_values("PPV_Abs", ascending=False)
    
    # Supplier pie (Top-N absolute contributors)
    sup_pie = ppv_sup.copy()
    sup_pie["PPV_Abs"] = sup_pie["PPV_Value"].abs()
    sup_pie["Sign"] = np.where(sup_pie["PPV_Value"] < 0, "Favorable", "Unfavorable")
    sup_pie = sup_pie[sup_pie["PPV_Abs"] > 0].sort_values("PPV_Abs", ascending=False)
    
    pie_left, pie_right = st.columns(2)
    with pie_left:
        st.caption("PPV value share by Item Category (absolute, sign in tooltip)")
        if not cat_pie.empty:
            fig_cat = px.pie(
                cat_pie, names="Item_Category", values="PPV_Abs", hole=0.35, title="PPV by Category"
            )
            fig_cat.update_traces(
                hovertemplate="**%{label}**<br>PPV ₹%{value:,.0f}<br>Sign: %{customdata}<extra></extra>",
                customdata=cat_pie["Sign"]
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No PPV data available for the selected period & filters.")
    with pie_right:
        st.caption(f"PPV value share by Supplier")
        if not sup_pie.empty:
            fig_sup = px.pie(
                sup_pie, names="Supplier", values="PPV_Abs", hole=0.35, title=f"PPV by Supplier"
            )
            fig_sup.update_traces(
                hovertemplate="**%{label}**<br>PPV ₹%{value:,.0f}<br>Sign: %{customdata}<extra></extra>",
                customdata=sup_pie["Sign"]
            )
            st.plotly_chart(fig_sup, use_container_width=True)
        else:
            st.info("No PPV data available for the selected period & filters.")
    
    # Tables (sign-aware, negative = favorable)
    c_left, c_right = st.columns(2)
    with c_left:
        st.caption("PPV by Category (negative = favorable)")
        st.dataframe(
            ppv_cat[["Item_Category", "PPV_Value", "PPV_Pct", "spend"]].round(2),
            use_container_width=True
        )
    with c_right:
        st.caption("PPV by Supplier (negative = favorable)")
        st.dataframe(
            ppv_sup[["Supplier", "PPV_Value", "PPV_Pct", "spend"]].round(2),
            use_container_width=True
        )
    
    
    # =============================================================
    # 💰 Savings & Working Capital (separate tab with Compare toggle)
    # =============================================================
    import re
    import plotly.express as px

    st.header("💰 Savings & Working Capital")
    st.caption("Compute Cost Reduction, Cost Avoidance, and Working Capital metrics for the selected period; optionally compare to the previous equal-length window.")

    # --- Helpers ---
    def _terms_to_days(s, invoice_date, due_date):
        """Map Payment_Terms like 'Net 30' or 'Advance' to days; fallback to due - invoice."""
        if isinstance(s, str):
            m = re.search(r"net\s*(\d+)", s, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
            if s.strip().lower().startswith("advance"):
                return 0
        if pd.notna(due_date) and pd.notna(invoice_date):
            return (due_date - invoice_date).days
        return np.nan

    def weighted_avg(series, weights):
        mask = series.notna() & weights.notna()
        s = series[mask]
        w = weights[mask]
        if len(s) == 0 or w.sum() == 0:
            return np.nan
        return float(np.average(s, weights=w))

    def _apply_filters_for_period(df, start_d, end_d):
        mask = pd.Series(True, index=df.index)
        if start_d:
            mask &= df["Invoice_Date"].dt.date >= start_d
        if end_d:
            mask &= df["Invoice_Date"].dt.date <= end_d
        # Reuse global selections from sidebar
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

    def _compute_metrics(df_slice, start_d, end_d):
        d = df_slice.copy()
        if d.empty:
            return {
                "CR": 0.0, "CA": 0.0, "DPO_actual": np.nan, "DPO_target": np.nan,
                "AP_float": 0.0, "spend": 0.0, "days": 1, "spend_per_day": 0.0,
                "CR_by_cat": pd.DataFrame(columns=["Item_Category","CR_Value"]),
                "CR_by_sup": pd.DataFrame(columns=["Supplier","CR_Value"]),
                "CA_by_cat": pd.DataFrame(columns=["Item_Category","CA_Value"]),
                "CA_by_sup": pd.DataFrame(columns=["Supplier","CA_Value"]),
                "monthly_dpo": pd.DataFrame(columns=["month","weighted_dpo"]) }

        # Savings
        
        d["cr_value"] = np.where(
            d["Unit_Price"].notna() & d["Negotiated_Price"].notna() &
            (d["Unit_Price"] < d["Negotiated_Price"]) &
            d["Quantity"].notna() & (d["Quantity"] > 0),
            round((d["Negotiated_Price"] - d["Unit_Price"]) * d["Quantity"],2),
            0.0
        )
        
        d["ca_value"] = np.where(
            d["Projected_Price"].notna() & d["Negotiated_Price"].notna() &
            (d["Projected_Price"] > d["Negotiated_Price"]) &
            d["Unit_Price"].notna() & (d["Unit_Price"] <= d["Negotiated_Price"]) &
            d["Quantity"].notna() & (d["Quantity"] > 0),
            round((d["Projected_Price"] - d["Negotiated_Price"]) * d["Quantity"],2),
            0.0
        )


        # Working capital
        d["terms_days"] = d.apply(lambda r: _terms_to_days(r.get("Payment_Terms"), r.get("Invoice_Date"), r.get("Invoice_Due_Date")), axis=1)
        d["dpo_actual"] = (d["Payment_Date"] - d["Invoice_Date"]).dt.days
        d["dpo_target"] = (d["Invoice_Due_Date"] - d["Invoice_Date"]).dt.days
        wa_dpo_actual = weighted_avg(d["dpo_actual"], d["Invoice_Amount"]) 
        wa_dpo_target = weighted_avg(d["dpo_target"], d["Invoice_Amount"]) 

        # Spend/day & AP float
        days_in_period = max((end_d - start_d).days, 1)
        spend_selected = float(d["Invoice_Amount"].sum())
        spend_per_day = spend_selected / days_in_period
        ap_float = spend_per_day * (wa_dpo_actual if pd.notna(wa_dpo_actual) else 0)

        # Aggregations
        cr_cat = d.groupby("Item_Category", dropna=False)["cr_value"].sum().sort_values(ascending=False)
        cr_sup = d.groupby("Supplier", dropna=False)["cr_value"].sum().sort_values(ascending=False)
        ca_cat = d.groupby("Item_Category", dropna=False)["ca_value"].sum().sort_values(ascending=False)
        ca_sup = d.groupby("Supplier", dropna=False)["ca_value"].sum().sort_values(ascending=False)

        # Monthly weighted DPO
        d["month"] = d["Invoice_Date"].dt.to_period("M").astype(str)
        def wavg_group(g):
            return weighted_avg(g["dpo_actual"], g["Invoice_Amount"]) 
        monthly_dpo = d.groupby("month").apply(wavg_group).rename("weighted_dpo").reset_index()
    
        return {
            "CR": round(float(d["cr_value"].sum()),2),
            "CA": round(float(d["ca_value"].sum()),2),
            "DPO_actual": wa_dpo_actual, "DPO_target": wa_dpo_target,
            "AP_float": float(ap_float), "spend": spend_selected,
            "days": days_in_period, "spend_per_day": float(spend_per_day),
            "CR_by_cat": cr_cat.reset_index().rename(columns={"cr_value":"CR_Value"}),
            "CR_by_sup": cr_sup.reset_index().rename(columns={"cr_value":"CR_Value"}),
            "CA_by_cat": ca_cat.reset_index().rename(columns={"ca_value":"CA_Value"}),
            "CA_by_sup": ca_sup.reset_index().rename(columns={"ca_value":"CA_Value"}),
            "monthly_dpo": monthly_dpo }

    # --- Inputs (use current sidebar period; add compare toggle here) ---
    compare_prev = st.checkbox("Compare to previous period", value=True)
    # Pull current period from existing sidebar control
    if isinstance(period, tuple):
        start_d, end_d = period
        if isinstance(start_d, pd.Timestamp): start_d = start_d.date()
        if isinstance(end_d, pd.Timestamp): end_d = end_d.date()
    else:
        start_d, end_d = period, date.today()

    # Current slice
    current_df = _apply_filters_for_period(base_df, start_d, end_d)
    current = _compute_metrics(current_df, start_d, end_d)
    

    # Previous slice
    previous = None
    prev_start = prev_end = None
    if compare_prev:
        prev_end = start_d - timedelta(days=1)
        prev_start = prev_end - (end_d - start_d)
        previous_df = _apply_filters_for_period(base_df, prev_start, prev_end)
        previous = _compute_metrics(previous_df, prev_start, prev_end)

    st.caption(f"Current period: **{start_d} → {end_d}**" + (f"  |  Previous: **{prev_start} → {prev_end}**" if compare_prev else ""))

    # KPI row with deltas
    k1, k2, k3, k4 = st.columns(4)
    def _delta(curr, prev):
        return None if prev is None or prev != prev else curr - prev

    if compare_prev and previous is not None:
        k1.metric("Cost Reduction (₹)", fmt_inr(current["CR"]), fmt_inr(_delta(current["CR"], previous["CR"])) )
        k2.metric("Cost Avoidance (₹)", fmt_inr(current["CA"]), fmt_inr(_delta(current["CA"], previous["CA"])) )
        dpo_delta = None if (previous["DPO_actual"] != previous["DPO_actual"]) or (current["DPO_actual"] != current["DPO_actual"]) else (current["DPO_actual"] - previous["DPO_actual"]) 
        k3.metric("Weighted DPO (days)", f"{current['DPO_actual']:.1f}" if current["DPO_actual"] == current["DPO_actual"] else "—",
                  f"{dpo_delta:+.1f}" if dpo_delta is not None else None)
        k4.metric("AP Float (₹)", fmt_inr(current["AP_float"]), fmt_inr(_delta(current["AP_float"], previous["AP_float"])) )
    else:
        k1.metric("Cost Reduction (₹)", fmt_inr(current["CR"]))
        k2.metric("Cost Avoidance (₹)", fmt_inr(current["CA"]))
        k3.metric("Weighted DPO (days)", f"{current['DPO_actual']:.1f}" if current["DPO_actual"] == current["DPO_actual"] else "—")
        k4.metric("AP Float (₹)", fmt_inr(current["AP_float"]))

    st.divider()

    cr_cat_curr = current["CR_by_cat"].copy(); cr_cat_curr["Period"] = "Current"
    if compare_prev and previous is not None:
        cr_cat_prev = previous["CR_by_cat"].copy()
        top_cats = cr_cat_curr["Item_Category"].tolist()
        cr_cat_prev = cr_cat_prev[cr_cat_prev["Item_Category"].isin(top_cats)]; cr_cat_prev["Period"] = "Previous"
        cr_cat_plot = pd.concat([cr_cat_curr, cr_cat_prev], ignore_index=True)
    else:
        cr_cat_plot = cr_cat_curr

    ca_cat_curr = current["CA_by_cat"]; ca_cat_curr["Period"] = "Current"
    if compare_prev and previous is not None:
        ca_cat_prev = previous["CA_by_cat"].copy()
        top_cats2 = ca_cat_curr["Item_Category"].tolist()
        ca_cat_prev = ca_cat_prev[ca_cat_prev["Item_Category"].isin(top_cats2)]; ca_cat_prev["Period"] = "Previous"
        ca_cat_plot = pd.concat([ca_cat_curr, ca_cat_prev], ignore_index=True)
    else:
        ca_cat_plot = ca_cat_curr

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Cost Reduction (CR) by Category")
        fig = px.bar(cr_cat_plot, x="Item_Category", y="CR_Value", color="Period", barmode="group", text="CR_Value")
        fig.update_layout(yaxis_title="CR (₹)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.caption("Cost Avoidance (CA) by Category")
        fig2 = px.bar(ca_cat_plot, x="Item_Category", y="CA_Value", color="Period", barmode="group", text="CA_Value")
        fig2.update_layout(yaxis_title="CA (₹)", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)
    
    if compare_prev and previous is not None:
        combined_CR_df = pd.concat(
            [current["CR_by_cat"].assign(Source="Current"),
             previous["CR_by_cat"].assign(Source="Previous")],
            ignore_index=True
        )
        combined_CA_df = pd.concat(
        [current["CA_by_cat"].assign(Source="Current"),
         previous["CA_by_cat"].assign(Source="Previous")],
        ignore_index=True
        )
    c_left, c_right = st.columns(2)
    with c_left:      
        if compare_prev and previous is not None:
            st.download_button("Download CR by Category (Previous & Current, CSV)", data=combined_CR_df.to_csv(index=False), file_name="cr_by_category_previous.csv", mime="text/csv")
        else:
            st.download_button("Download CR by Category (Current, CSV)", data=current["CR_by_cat"].to_csv(index=False), file_name="cr_by_category_current.csv", mime="text/csv")
    with c_right:
        if compare_prev and previous is not None:
            st.download_button("Download CA by Category (Previous & Current, CSV)", data=combined_CA_df.to_csv(index=False), file_name="ca_by_category_previous.csv", mime="text/csv")
        else:
            st.download_button("Download CA by Category (Current, CSV)", data=current["CA_by_cat"].to_csv(index=False), file_name="ca_by_category_current.csv", mime="text/csv")
    st.divider()

    cr_sup_curr = current["CR_by_sup"].copy(); cr_sup_curr["Period"] = "Current"
    if compare_prev and previous is not None:
        cr_sup_prev = previous["CR_by_sup"].copy(); top_sups = cr_sup_curr["Supplier"].tolist()
        cr_sup_prev = cr_sup_prev[cr_sup_prev["Supplier"].isin(top_sups)]; cr_sup_prev["Period"] = "Previous"
        cr_sup_plot = pd.concat([cr_sup_curr, cr_sup_prev], ignore_index=True)
    else:
        cr_sup_plot = cr_sup_curr

    ca_sup_curr = current["CA_by_sup"].copy(); ca_sup_curr["Period"] = "Current"
    if compare_prev and previous is not None:
        ca_sup_prev = previous["CA_by_sup"].copy(); top_sups2 = ca_sup_curr["Supplier"].tolist()
        ca_sup_prev = ca_sup_prev[ca_sup_prev["Supplier"].isin(top_sups2)]; ca_sup_prev["Period"] = "Previous"
        ca_sup_plot = pd.concat([ca_sup_curr, ca_sup_prev], ignore_index=True)
    else:
        ca_sup_plot = ca_sup_curr

    s1, s2 = st.columns(2)
    with s1:
        st.caption("Cost Reduction (CR) by Supplier")
        fig3 = px.bar(cr_sup_plot, x="Supplier", y="CR_Value", color="Period", barmode="group", text="CR_Value")
        fig3.update_layout(yaxis_title="CR (₹)", xaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)
    with s2:
        st.caption("Cost Avoidance (CA) by Supplier")
        fig4 = px.bar(ca_sup_plot, x="Supplier", y="CA_Value", color="Period", barmode="group", text="CA_Value")
        fig4.update_layout(yaxis_title="CA (₹)", xaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    if compare_prev and previous is not None:
        combined_CR_sup_df = pd.concat(
            [current["CR_by_sup"].assign(Source="Current"),
             previous["CR_by_sup"].assign(Source="Previous")],
            ignore_index=True
        )
        combined_CA_sup_df = pd.concat(
        [current["CA_by_sup"].assign(Source="Current"),
         previous["CA_by_sup"].assign(Source="Previous")],
        ignore_index=True
        )
    c_left, c_right = st.columns(2)
    with c_left:   
        if compare_prev and previous is not None:
            st.download_button("Download CR by Supplier (Previous, CSV)", data=combined_CR_sup_df.to_csv(index=False), file_name="cr_by_supplier_previous.csv", mime="text/csv")
        else:
            st.download_button("Download CR by Supplier (Current, CSV)", data=current["CR_by_sup"].to_csv(index=False), file_name="cr_by_supplier_current.csv", mime="text/csv")
    with c_right:
        if compare_prev and previous is not None:
            st.download_button("Download CA by Supplier (Previous, CSV)", data=combined_CA_sup_df.to_csv(index=False), file_name="ca_by_supplier_previous.csv", mime="text/csv")
        else:
            st.download_button("Download CA by Supplier (Current, CSV)", data=current["CA_by_sup"].to_csv(index=False), file_name="ca_by_supplier_current.csv", mime="text/csv")
    st.divider()

    # Working Capital: Monthly DPO
    st.subheader("Working Capital: Monthly Weighted DPO")
    monthly_curr = current["monthly_dpo"]; monthly_curr["Period"] = "Current"
    if compare_prev and previous is not None and not previous["monthly_dpo"].empty:
        monthly_prev = previous["monthly_dpo"].copy(); monthly_prev["Period"] = "Previous"
        monthly_both = pd.concat([monthly_curr, monthly_prev], ignore_index=True)
    else:
        monthly_both = monthly_curr
    fig5 = px.line(monthly_both, x="month", y="weighted_dpo", color="Period", markers=True,
                   labels={"weighted_dpo":"Weighted DPO (days)", "month":""})
    st.plotly_chart(fig5, use_container_width=True)

    # Details table
    st.subheader("DPO Details (Current Period)")
    dpo_tbl = current_df[["Supplier","Invoice_Number","Invoice_Date","Invoice_Amount","Invoice_Due_Date","Payment_Date"]].copy()
    dpo_tbl["dpo_actual"] = (dpo_tbl["Payment_Date"] - dpo_tbl["Invoice_Date"]).dt.days
    dpo_tbl["dpo_target"] = (dpo_tbl["Invoice_Due_Date"] - dpo_tbl["Invoice_Date"]).dt.days
    st.dataframe(dpo_tbl.sort_values("Invoice_Date", ascending=False), use_container_width=True, height=360)
    st.download_button("Download DPO detail (Current, CSV)", data=dpo_tbl.to_csv(index=False), file_name="dpo_detail_current.csv", mime="text/csv")

    with st.expander("Definitions & Notes"):
        st.markdown("""
- **Cost Reduction (CR):** Realized savings when Unit_Price < Negotiated_Price.
- **Cost Avoidance (CA):** Avoided price increase where Projected_Price > Negotiated_Price and actual paid ≤ Negotiated_Price.
- **Weighted DPO:** Weighted average of (Payment_Date − Invoice_Date) by Invoice_Amount.
- **AP Float (proxy):** Spend/day × Weighted DPO. Directional indicator of AP-related working capital.
- **Comparison window:** Previous period is equal length and ends the day before the current period starts; all other filters are identical.
- **Edge cases:** Missing dates and zeros are ignored in weighted calculations; results may differ slightly when data is sparse.
        """)


# =============================================================
# 📈 Time-series Forecast (SARIMA only): Category Spend (Next Quarter)
# =============================================================
import streamlit as st


with tabs[1]:
    st.header("📈 Time-series Forecast (SARIMA): Category Spend for Next Quarter")
    st.caption("Forecast monthly spend by Item Category for the next quarter. View results in a table and download as CSV.")
    # Choose training data slice
    df_input = base_df.copy()
    
    # Controls
    horizon = st.slider("Forecast horizon (months)", 1, 12, 3, step=1)
    #season = st.slider("Seasonal period (months)", 12, 24, 12, step=1)
    #min_points = st.slider("Minimum historical months per category", 3, 36, 6, step=1)
    exclude_cancelled = st.checkbox("Exclude 'Cancelled' invoices", value=True)
    
    # Run forecast (SARIMA-only with seasonal-naive fallback)
    fc_ts_simple = forecast_by_category_timeseries_simple(
        df_input, horizon=horizon, season=12, min_points=36, exclude_cancelled=exclude_cancelled
    )
    
    # Show
    st.dataframe(fc_ts_simple, use_container_width=True)
    
    # Download CSV
    st.download_button(
        label="Download Forecast by Category (CSV)",
        data=fc_ts_simple.to_csv(index=False),
        file_name=f"forecast_by_category_sarima_next_{horizon}_months.csv",
        mime="text/csv"
    )
    
    with st.expander("Notes"):
        st.markdown("""
    - **Model:** SARIMA chosen via a small AIC grid. If the category history is too short or the fit fails, we use **Seasonal-Naive**.
    - **Confidence intervals:** 80% and 95% from SARIMA; for the fallback we estimate bands using residual volatility vs seasonal lag.
    - **Rounding:** All values are rounded to **2 decimals**.
    """)

with tabs[2]:
    st.header("🤝 Supplier Optimization")
    st.caption("Recommend the best supplier per category using cost, reliability, risk, and volume fit. Weights are tunable.")

    # --- Inputs ---
    wcost = 40 #st.slider("Weight: Cost (PPV)", 0, 100, 40, step=5)
    wrel  = 30 #st.slider("Weight: Reliability (OTD)", 0, 100, 30, step=5)
    wrisk = 20 #st.slider("Weight: Risk (discrepancy & late)", 0, 100, 20, step=5)
    wvol  = 10 #st.slider("Weight: Volume fit (spend & qty)", 0, 100, 10, step=5)
    cost_mix_PPV_Pct = 70 #st.slider("Cost mix: PPV% vs PPV value (PPV% weight)", 0, 100, 70, step=5)

    weights = {
        "cost": wcost/100.0,
        "reliability": wrel/100.0,
        "risk": wrisk/100.0,
        "volume": wvol/100.0
    }

    # Data slice: use CURRENT filtered selection
    d_in = filtered.copy()

    # --- Compute KPIs & score ---
    kpis = compute_supplier_kpis(d_in)
    scored = score_suppliers(kpis, weights, cost_mix_PPV_Pct=cost_mix_PPV_Pct/100.0)

    # --- Recommendation per category (top by Supplier_Score) ---
    st.subheader("Recommended Supplier per Category")
    recs = (
        scored.sort_values(["Item_Category","Supplier_Score"], ascending=[True, False])
              .groupby("Item_Category")
              .head(1)
              .copy()
    )

    # Pretty columns & rounding
    show_cols = [
        "Item_Category","Supplier","Supplier_Score",
        "spend","qty","txn_count","PPV_Pct","PPV_Value","otd_rate","disc_rate","late_rate","Rationale"
    ]
    recs_view = recs[show_cols].copy()
    recs_view["Supplier_Score"] = recs_view["Supplier_Score"].round(2)
    recs_view["spend"] = recs_view["spend"].round(2)
    recs_view["qty"] = recs_view["qty"].round(2)
    recs_view["PPV_Pct"] = recs_view["PPV_Pct"].round(2)
    recs_view["PPV_Value"] = recs_view["PPV_Value"].round(2)
    recs_view["otd_rate"] = (recs_view["otd_rate"]*100.0).round(1)
    recs_view["disc_rate"] = (recs_view["disc_rate"]*100.0).round(1)
    recs_view["late_rate"] = (recs_view["late_rate"]*100.0).round(1)
    recs_view = recs_view.rename(columns={
        "Supplier_Score": "Score (0–100)",
        "spend": "Spend (₹)",
        "qty": "Qty",
        "PPV_Pct": "PPV (%)",
        "PPV_Value": "PPV Value (₹)",
        "otd_rate": "OTD (%)",
        "disc_rate": "Discrepancy (%)",
        "late_rate": "Late payments (%)"
    })

    st.dataframe(recs_view, use_container_width=True)

    st.download_button(
        "Download recommendations (CSV)",
        data=recs_view.to_csv(index=False),
        file_name="supplier_recommendations_by_category.csv",
        mime="text/csv"
    )

    st.divider()

    # --- Drilldown ranking for a selected category ---
    if not scored.empty:
        cat_sel = st.selectbox(
            "Drilldown: choose a category",
            options=sorted(scored["Item_Category"].dropna().unique().tolist())
        )

        if cat_sel:
            drill = scored[scored["Item_Category"] == cat_sel].copy()
            drill = drill.sort_values("Supplier_Score", ascending=False)

            st.caption(f"Supplier ranking for '{cat_sel}' (0–100)")
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=drill["Supplier"],
                y=drill["Supplier_Score"],
                text = drill["Supplier_Score"],
                textposition="outside"
            ))
            fig.update_layout(yaxis_title="Score (0–100)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            # Download drilldown
            drill_view = drill[show_cols].copy()
            drill_view["Supplier_Score"] = drill_view["Supplier_Score"].round(2)
            drill_view["spend"] = drill_view["spend"].round(2)
            drill_view["qty"] = drill_view["qty"].round(2)
            drill_view["PPV_Pct"] = drill_view["PPV_Pct"].round(2)
            drill_view["PPV_Value"] = drill_view["PPV_Value"].round(2)
            drill_view["otd_rate"] = (drill_view["otd_rate"]*100.0).round(1)
            drill_view["disc_rate"] = (drill_view["disc_rate"]*100.0).round(1)
            drill_view["late_rate"] = (drill_view["late_rate"]*100.0).round(1)
            drill_view = drill_view.rename(columns={
                "Supplier_Score": "Score (0–100)",
                "spend": "Spend (₹)",
                "qty": "Qty",
                "PPV_Pct": "PPV (%)",
                "PPV_Value": "PPV Value (₹)",
                "otd_rate": "OTD (%)",
                "disc_rate": "Discrepancy (%)",
                "late_rate": "Late payments (%)"
            })


    with st.expander("Scoring methodology"):
        st.markdown("""
    **Sub-scores (normalized per category):**
    - **Cost** = blend of *inverted* PPV% and *inverted* unfavorable PPV value (positives only),
    - **Reliability** = normalized On-Time Delivery rate (higher is better),
    - **Risk** = inverted normalized composite of discrepancy & late-payment rates,
    - **Volume** = normalized mix of historical spend & quantity.
    
    **Final Score (0–100)** = weighted sum of sub-scores using your sliders.
    Default weights: **Cost 40%**, **Reliability 30%**, **Risk 20%**, **Volume 10%**.
    """)
        
with tabs[3]:
    model = fit_memory(base_df)
    st.header("🗂️ Auto Categorize")
    st.caption("Auto categorizes a given product into its category.  Product description and supplier to be selected.")
    
    prod = st.text_input("Product name (Item_Description)")
    sup = st.selectbox("Supplier", options=sorted(base_df["Supplier"].dropna().unique()))
    
    buttonpressed = st.button("Categorize the product")
    if buttonpressed:
        pr = predict(model, prod, sup, top_k=20)
        st.json(pr)
