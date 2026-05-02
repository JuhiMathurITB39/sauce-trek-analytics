"""
app/streamlit_app.py
--------------------
Sauce Trek — Customer Retention & Revenue Intelligence Dashboard
Run: streamlit run app/streamlit_app.py
"""

import sys
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent
MASTER_CSV = BASE_DIR / "outputs" / "reports" / "customer_master.csv"
REPORT_MD  = BASE_DIR / "outputs" / "reports" / "insights_report.md"
sys.path.insert(0, str(BASE_DIR))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sauce Trek Analytics",
    page_icon="🥫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px; padding: 1rem 1.4rem;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #a0aab8; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: #ffffff; font-size: 1.8rem; margin: 4px 0 0 0; font-weight: 700; }
    .metric-card p  { color: #6b7280; font-size: 0.75rem; margin: 2px 0 0 0; }
    .section-header { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin: 1rem 0 0.5rem 0; border-bottom: 1px solid #2d3250; padding-bottom: 6px; }
    .insight-box { background: #1a1f35; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid #3b82f6; }
    .stMetric label { font-size: 0.8rem !important; color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Colour map ────────────────────────────────────────────────────────────────
SEG_COLORS = {
    "Champions":          "#10b981",
    "Loyal Customers":    "#3b82f6",
    "Promising":          "#f59e0b",
    "Discount Dependent": "#f97316",
    "At Risk":            "#ef4444",
    "Lost / Inactive":    "#6b7280",
    "Need Attention":     "#8b5cf6",
}
RISK_COLORS = {"Low Risk": "#10b981", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        st.error("Customer master CSV not found. Please run `python run_analysis.py` first.")
        st.stop()
    df = pd.read_csv(MASTER_CSV)
    # ensure types
    for col in ["recency_days","frequency","monetary","churn_prob","ltv_predicted"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def kpi_card(col, title: str, value: str, subtitle: str = "", color: str = "#3b82f6"):
    col.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
        <h3>{title}</h3>
        <h2>{value}</h2>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df_all = load_master()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🥫 Sauce Trek Analytics")
        st.markdown("**Customer Intelligence Dashboard**")
        st.divider()

        cities = ["All"] + sorted(df_all["city"].dropna().unique().tolist())
        sel_city = st.selectbox("Filter by City", cities)

        segments = ["All"] + sorted(df_all["segment_label"].dropna().unique().tolist())
        sel_seg  = st.selectbox("Filter by Segment", segments)

        risk_tiers = ["All"] + ["Low Risk","Medium Risk","High Risk"]
        sel_risk   = st.selectbox("Filter by Churn Risk", risk_tiers)

        min_rev, max_rev = int(df_all["monetary"].min()), int(df_all["monetary"].max())
        rev_range = st.slider("Revenue Range (₹)", min_rev, max_rev, (min_rev, max_rev))

        st.divider()
        st.markdown(f"**Dataset:** {len(df_all):,} customers")
        st.markdown("**Ref. date:** 30 Jun 2024")

    # ── Apply filters ─────────────────────────────────────────────────────
    df = df_all.copy()
    if sel_city  != "All": df = df[df["city"]           == sel_city]
    if sel_seg   != "All": df = df[df["segment_label"]  == sel_seg]
    if sel_risk  != "All": df = df[df["churn_risk_tier"]== sel_risk]
    df = df[(df["monetary"] >= rev_range[0]) & (df["monetary"] <= rev_range[1])]

    if df.empty:
        st.warning("No customers match the selected filters.")
        return

    # ─────────────────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("# 🥫 Sauce Trek — Customer Intelligence Dashboard")
    st.markdown(f"*Showing **{len(df):,}** of **{len(df_all):,}** customers*")
    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # KPI ROW
    # ─────────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_rev  = df["monetary"].sum()
    avg_aov    = df["avg_order_value"].mean()
    churn_rate = df["is_churned"].mean() * 100
    high_risk  = (df["churn_risk_tier"] == "High Risk").sum()
    avg_ltv    = df["ltv_predicted"].mean() if "ltv_predicted" in df.columns else 0
    avg_rec    = df["recency_days"].mean()

    kpi_card(c1, "Total Revenue",    f"₹{total_rev/1e6:.1f}M",  f"{len(df):,} customers", "#10b981")
    kpi_card(c2, "Avg Order Value",  f"₹{avg_aov:,.0f}",        "per order", "#3b82f6")
    kpi_card(c3, "Churn Rate",       f"{churn_rate:.1f}%",       "60-day definition", "#ef4444")
    kpi_card(c4, "High Risk",        f"{high_risk:,}",           "customers at risk", "#f97316")
    kpi_card(c5, "Avg Pred. LTV",    f"₹{avg_ltv:,.0f}",        "12-month horizon", "#8b5cf6")
    kpi_card(c6, "Avg Recency",      f"{avg_rec:.0f}d",          "since last purchase", "#f59e0b")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # ROW 2: Segments + Churn Risk
    # ─────────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<p class="section-header"> Customer Segments</p>', unsafe_allow_html=True)
        seg_data = df.groupby("segment_label").agg(
            customers=("customer_id","count"),
            revenue=("monetary","sum"),
        ).reset_index().sort_values("revenue", ascending=False)
        seg_data["revenue_share"] = seg_data["revenue"] / seg_data["revenue"].sum() * 100
        seg_data["color"] = seg_data["segment_label"].map(SEG_COLORS)

        fig_seg = px.bar(
            seg_data, x="segment_label", y="revenue",
            color="segment_label",
            color_discrete_map=SEG_COLORS,
            text=seg_data["revenue_share"].apply(lambda x: f"{x:.1f}%"),
            labels={"revenue":"Total Revenue (₹)","segment_label":"Segment"},
            template="plotly_dark",
        )
        fig_seg.update_traces(textposition="outside", textfont_size=11)
        fig_seg.update_layout(
            showlegend=False, margin=dict(t=10,b=10,l=0,r=0),
            height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header"> Churn Risk Distribution</p>', unsafe_allow_html=True)
        risk_data = df["churn_risk_tier"].value_counts().reset_index()
        risk_data.columns = ["tier","count"]
        risk_data["color"] = risk_data["tier"].map(RISK_COLORS)

        fig_risk = px.pie(
            risk_data, names="tier", values="count",
            color="tier", color_discrete_map=RISK_COLORS,
            template="plotly_dark", hole=0.45,
        )
        fig_risk.update_traces(textposition="inside", textinfo="percent+label",
                                textfont_size=11)
        fig_risk.update_layout(
            showlegend=True, margin=dict(t=10,b=10,l=0,r=0),
            height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # ROW 3: RFM Scatter + Pareto
    # ─────────────────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-header"> RFM Scatter Map</p>', unsafe_allow_html=True)
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig_rfm = px.scatter(
            sample,
            x="recency_days", y="frequency",
            size=np.clip(sample["monetary"], 0, sample["monetary"].quantile(0.99)),
            color="segment_label",
            color_discrete_map=SEG_COLORS,
            opacity=0.6,
            template="plotly_dark",
            labels={"recency_days":"Recency (days)","frequency":"Purchase Frequency"},
            size_max=25,
        )
        fig_rfm.update_layout(
            margin=dict(t=10,b=10,l=0,r=0), height=320,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_rfm, use_container_width=True)

    with col4:
        st.markdown('<p class="section-header"> Revenue Concentration (Pareto)</p>', unsafe_allow_html=True)
        pareto = df.sort_values("monetary", ascending=False).copy()
        pareto["cum_rev_pct"] = pareto["monetary"].cumsum() / pareto["monetary"].sum() * 100
        pareto["cum_cust_pct"] = (np.arange(1, len(pareto)+1) / len(pareto)) * 100

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Scatter(
            x=pareto["cum_cust_pct"], y=pareto["cum_rev_pct"],
            mode="lines", name="Revenue Curve",
            line=dict(color="#3b82f6", width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
        ))
        fig_pareto.add_trace(go.Scatter(
            x=[0,100], y=[0,100], mode="lines", name="Equal (45°)",
            line=dict(color="#6b7280", dash="dash", width=1)
        ))
        # 20% line
        idx = (pareto["cum_cust_pct"] - 20).abs().idxmin()
        rev_at_20 = pareto.loc[idx, "cum_rev_pct"]
        fig_pareto.add_shape(type="line", x0=20, x1=20, y0=0, y1=rev_at_20,
                              line=dict(color="#ef4444", dash="dot", width=1.5))
        fig_pareto.add_annotation(x=20, y=rev_at_20,
                                   text=f"Top 20%→{rev_at_20:.0f}% rev",
                                   showarrow=True, arrowhead=2, arrowcolor="#ef4444",
                                   font=dict(color="#ef4444", size=10))
        fig_pareto.update_layout(
            template="plotly_dark", margin=dict(t=10,b=10,l=0,r=0), height=320,
            xaxis_title="Cumulative % Customers", yaxis_title="Cumulative % Revenue",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # ROW 4: Churn Prob Histogram + LTV by Segment
    # ─────────────────────────────────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<p class="section-header"> Churn Probability Distribution</p>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        for label, color, name in [(0,"#10b981","Active"),(1,"#ef4444","Churned")]:
            subset = df[df["is_churned"]==label]["churn_prob"]
            fig_hist.add_trace(go.Histogram(
                x=subset, name=name, marker_color=color,
                opacity=0.65, nbinsx=40
            ))
        fig_hist.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1,
                           yref="paper", line=dict(color="white", dash="dash", width=1.5))
        fig_hist.update_layout(
            barmode="overlay", template="plotly_dark",
            margin=dict(t=10,b=10,l=0,r=0), height=300,
            xaxis_title="Churn Probability", yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col6:
        st.markdown('<p class="section-header"> Predicted LTV by Segment</p>', unsafe_allow_html=True)
        if "ltv_predicted" in df.columns:
            ltv_data = df.groupby("segment_label")["ltv_predicted"].median().sort_values(ascending=False).reset_index()
            fig_ltv = px.bar(
                ltv_data, x="segment_label", y="ltv_predicted",
                color="segment_label", color_discrete_map=SEG_COLORS,
                template="plotly_dark",
                labels={"ltv_predicted":"Median Predicted LTV (₹)","segment_label":"Segment"}
            )
            fig_ltv.update_layout(
                showlegend=False, margin=dict(t=10,b=10,l=0,r=0), height=300,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_ltv, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # ROW 5: Discount Analysis
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header"> Discount Dependency Analysis</p>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)

    with col7:
        disc_seg = df.groupby("segment_label")["discount_usage_rate"].mean().sort_values(ascending=False).reset_index()
        fig_disc = px.bar(
            disc_seg, x="discount_usage_rate", y="segment_label",
            orientation="h", color="segment_label",
            color_discrete_map=SEG_COLORS, template="plotly_dark",
            labels={"discount_usage_rate":"Avg Discount Usage Rate (%)","segment_label":"Segment"}
        )
        fig_disc.update_layout(
            showlegend=False, margin=dict(t=10,b=10,l=0,r=0), height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_disc, use_container_width=True)

    with col8:
        # discount rate vs revenue scatter
        sample2 = df.sample(min(1500, len(df)), random_state=1)
        fig_dsc2 = px.scatter(
            sample2, x="discount_usage_rate", y="monetary",
            color="segment_label", color_discrete_map=SEG_COLORS,
            template="plotly_dark", opacity=0.5,
            trendline="ols",
            labels={"discount_usage_rate":"Discount Usage Rate (%)","monetary":"Total Revenue (₹)"}
        )
        fig_dsc2.update_layout(
            showlegend=False, margin=dict(t=10,b=10,l=0,r=0), height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dsc2, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # ROW 6: Customer Table (filterable)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🔍 Customer Explorer</p>', unsafe_allow_html=True)
    display_cols = [
        c for c in [
            "customer_id","city","segment_label","churn_risk_tier",
            "recency_days","frequency","monetary","avg_order_value",
            "discount_usage_rate","churn_prob","ltv_predicted","ltv_tier"
        ] if c in df.columns
    ]
    st.dataframe(
        df[display_cols].sort_values("ltv_predicted", ascending=False).head(200),
        use_container_width=True,
        height=380,
    )

    # ─────────────────────────────────────────────────────────────────────
    # ROW 7: Segment Summary Table
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header"> Segment Summary</p>', unsafe_allow_html=True)
    total_rev_all = df["monetary"].sum()
    seg_table = df.groupby("segment_label").agg(
        Customers=("customer_id","count"),
        Avg_Recency=("recency_days","mean"),
        Avg_Orders=("frequency","mean"),
        Avg_Revenue=("monetary","mean"),
        Total_Revenue=("monetary","sum"),
        Avg_Discount_Rate=("discount_usage_rate","mean"),
        Avg_Churn_Prob=("churn_prob","mean"),
        Avg_LTV=("ltv_predicted","mean"),
    ).round(1).reset_index()
    seg_table["Revenue_Share_%"] = (seg_table["Total_Revenue"] / total_rev_all * 100).round(1)
    st.dataframe(seg_table.sort_values("Total_Revenue", ascending=False), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # Insights Report Tab
    # ─────────────────────────────────────────────────────────────────────
    with st.expander(" View Full Consulting Insights Report"):
        if REPORT_MD.exists():
            st.markdown(REPORT_MD.read_text(encoding="utf-8"))
        else:
            st.info("Run `python run_analysis.py` to generate the report.")


if __name__ == "__main__":
    main()
