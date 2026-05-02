"""
src/ltv_model.py
-----------------
Customer Lifetime Value (LTV) Estimation
  - BG/NBD-style heuristic LTV using observed RFM
  - Gradient Boosting regression for future-value prediction
  - Cohort retention curve visualisation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, r2_score
from sklearn.preprocessing   import StandardScaler

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_HORIZON_DAYS = 365   # predict LTV over next 12 months


# ─────────────────────────────────────────────────────────────────────────────
# 1. Heuristic LTV (interpretable baseline)
# ─────────────────────────────────────────────────────────────────────────────

def compute_heuristic_ltv(df: pd.DataFrame, horizon_days: int = PREDICTION_HORIZON_DAYS) -> pd.DataFrame:
    """
    Simple but defensible LTV formula:
      LTV = avg_order_value × purchase_rate_per_day × horizon × retention_factor
    where retention_factor ∈ (0,1) is calibrated from recency.
    """
    out = df.copy()

    # daily purchase rate (orders per day in active window)
    out["active_days"] = out["lifespan_days"].clip(lower=1)
    out["daily_purchase_rate"] = out["frequency"] / out["active_days"]

    # retention factor: exponential decay based on recency
    # half-life ≈ 90 days (typical for FMCG)
    half_life = 90.0
    out["retention_factor"] = np.exp(-np.log(2) * out["recency_days"] / half_life)

    # discount adjustment (higher discount dependency → lower net LTV)
    out["discount_ltv_haircut"] = 1 - (out["discount_usage_rate"] / 100) * 0.25

    # heuristic LTV
    out["ltv_heuristic"] = (
        out["avg_order_value"]
        * out["daily_purchase_rate"]
        * horizon_days
        * out["retention_factor"]
        * out["discount_ltv_haircut"]
    ).round(2)

    # floor at 0
    out["ltv_heuristic"] = out["ltv_heuristic"].clip(lower=0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. ML LTV model (GBR)
# ─────────────────────────────────────────────────────────────────────────────

LTV_FEATURES = [
    "recency_days", "frequency", "monetary",
    "avg_order_value", "discount_usage_rate",
    "orders_per_month", "lifespan_days",
    "recent_spend_ratio", "distinct_products_bought",
    "mayo_spend_share", "tenure_days",
]

def train_ltv_model(df: pd.DataFrame):
    """
    Train a GBR to predict heuristic_ltv (proxy for future value).
    In production, replace target with actual observed 12-month future revenue.
    """
    df = df.copy()
    feats = [f for f in LTV_FEATURES if f in df.columns]
    target = "ltv_heuristic"

    X = df[feats].fillna(0).values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=4,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"  LTV Model — MAE: ₹{mae:.0f}   R²: {r2:.4f}")

    return model, feats, mae, r2


def score_ltv(df: pd.DataFrame, model, feats: list) -> pd.DataFrame:
    """Add predicted_ltv column to dataframe."""
    X = df[feats].fillna(0).values
    df = df.copy()
    df["ltv_predicted"] = model.predict(X).clip(min=0).round(2)
    df["ltv_tier"] = pd.qcut(
        df["ltv_predicted"],
        q=4,
        labels=["Bronze", "Silver", "Gold", "Platinum"]
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cohort retention chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_retention_heatmap(cohort_df: pd.DataFrame, save: bool = True):
    """Classic cohort retention heatmap."""
    if cohort_df.empty:
        return None

    pivot = cohort_df.pivot_table(
        index="cohort_month",
        columns="period_number",
        values="retention_rate_pct",
    )
    # keep max 18 periods and last 24 cohorts
    pivot = pivot.iloc[-24:, :19]

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        pivot,
        annot=True, fmt=".0f",
        cmap="YlGn",
        linewidths=0.4,
        linecolor="white",
        vmin=0, vmax=100,
        cbar_kws={"label": "Retention %"},
        ax=ax,
        annot_kws={"fontsize": 7},
    )
    ax.set_title("Monthly Cohort Retention Heatmap (%)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Months Since First Purchase", fontsize=11)
    ax.set_ylabel("Acquisition Cohort", fontsize=11)
    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "09_cohort_retention_heatmap.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '09_cohort_retention_heatmap.png'}")
    return fig


def plot_avg_retention_curve(cohort_df: pd.DataFrame, save: bool = True):
    """Average retention curve across all cohorts."""
    if cohort_df.empty:
        return None
    avg = cohort_df.groupby("period_number")["retention_rate_pct"].mean().reset_index()
    avg = avg[avg["period_number"] <= 12]   # 12-month window

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(avg["period_number"], avg["retention_rate_pct"], alpha=0.2, color="#3498DB")
    ax.plot(avg["period_number"], avg["retention_rate_pct"],
            color="#2C3E50", lw=2.5, marker="o", markersize=6)
    # annotate key months
    for _, row in avg.iterrows():
        if row["period_number"] in [0, 1, 3, 6, 12]:
            ax.annotate(
                f"{row['retention_rate_pct']:.0f}%",
                (row["period_number"], row["retention_rate_pct"]),
                textcoords="offset points", xytext=(4, 6), fontsize=9, fontweight="bold"
            )
    ax.set_xlabel("Months Since First Purchase", fontsize=11)
    ax.set_ylabel("Average Retention Rate (%)", fontsize=11)
    ax.set_title("Average Customer Retention Curve\n(All Cohorts)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xticks(avg["period_number"])
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "10_avg_retention_curve.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '10_avg_retention_curve.png'}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. LTV visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_ltv_by_segment(df: pd.DataFrame, save: bool = True):
    """Box plot of predicted LTV by segment label."""
    if "segment_label" not in df.columns:
        return None

    order = df.groupby("segment_label")["ltv_predicted"].median().sort_values(ascending=False).index
    palette = {
        "Champions":"#2ECC71","Loyal Customers":"#3498DB","Promising":"#F39C12",
        "Discount Dependent":"#E67E22","At Risk":"#E74C3C",
        "Lost / Inactive":"#95A5A6","Need Attention":"#9B59B6"
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df, x="segment_label", y="ltv_predicted",
        order=order, palette=palette, ax=ax,
        flierprops={"marker":"o","markersize":3,"alpha":0.4}
    )
    ax.set_xlabel("Segment", fontsize=11)
    ax.set_ylabel("Predicted 12-Month LTV (₹)", fontsize=11)
    ax.set_title("Predicted LTV Distribution by Segment", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "11_ltv_by_segment.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '11_ltv_by_segment.png'}")
    return fig


def plot_ltv_scatter(df: pd.DataFrame, save: bool = True):
    """Actual vs predicted LTV scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["ltv_heuristic"], df["ltv_predicted"], alpha=0.25, s=10,
               color="#3498DB", edgecolors="none")
    lim = max(df["ltv_heuristic"].quantile(0.99), df["ltv_predicted"].quantile(0.99))
    ax.plot([0, lim],[0, lim],"r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Heuristic LTV (₹)", fontsize=11)
    ax.set_ylabel("ML Predicted LTV (₹)", fontsize=11)
    ax.set_title("LTV: Heuristic vs ML Prediction", fontsize=13, fontweight="bold")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "12_ltv_scatter.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '12_ltv_scatter.png'}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Master function
# ─────────────────────────────────────────────────────────────────────────────

def run_ltv_analysis(df: pd.DataFrame, cohort_df: pd.DataFrame, verbose: bool = True):
    print("💰  Running LTV analysis...")

    # heuristic LTV
    df = compute_heuristic_ltv(df)

    # ML LTV
    ltv_model, feats, mae, r2 = train_ltv_model(df)
    df = score_ltv(df, ltv_model, feats)

    if verbose:
        tier_summary = df.groupby("ltv_tier").agg(
            count=("customer_id","count"),
            avg_ltv=("ltv_predicted","mean"),
            total_ltv=("ltv_predicted","sum")
        ).round(0)
        print("\n  LTV Tier Summary:")
        print(tier_summary.to_string())

    # charts
    plot_retention_heatmap(cohort_df)
    plot_avg_retention_curve(cohort_df)
    plot_ltv_by_segment(df)
    plot_ltv_scatter(df)

    print("  ✅  LTV analysis complete.\n")
    return df, ltv_model
