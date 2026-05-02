"""
src/segmentation.py
--------------------
Customer Segmentation Engine
  1. RFM percentile scoring
  2. K-Means clustering (k=5)
  3. Segment labelling & business interpretation
  4. Pareto / revenue contribution analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    "Champions":         "#2ECC71",
    "Loyal Customers":   "#3498DB",
    "Promising":         "#F39C12",
    "Discount Dependent":"#E67E22",
    "At Risk":           "#E74C3C",
    "Lost / Inactive":   "#95A5A6",
    "Need Attention":    "#9B59B6",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. RFM Scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  ml_features DataFrame
    Output: same DF with r_score, f_score, m_score, rfm_total added.
    Scoring: 1 (worst) → 5 (best) using percentile breaks.
    Recency is inverted (lower days = better).
    """
    out = df.copy()

    # Recency score — lower recency_days = higher score
    r_breaks = out["recency_days"].quantile([0.20, 0.40, 0.60, 0.80]).values
    out["r_score"] = pd.cut(
        out["recency_days"],
        bins=[-np.inf, r_breaks[0], r_breaks[1], r_breaks[2], r_breaks[3], np.inf],
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # Frequency score — higher = better
    f_breaks = out["frequency"].quantile([0.20, 0.40, 0.60, 0.80]).values
    out["f_score"] = pd.cut(
        out["frequency"],
        bins=[-np.inf, f_breaks[0], f_breaks[1], f_breaks[2], f_breaks[3], np.inf],
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Monetary score — higher = better
    m_breaks = out["monetary"].quantile([0.20, 0.40, 0.60, 0.80]).values
    out["m_score"] = pd.cut(
        out["monetary"],
        bins=[-np.inf, m_breaks[0], m_breaks[1], m_breaks[2], m_breaks[3], np.inf],
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    out["rfm_total"] = out["r_score"] + out["f_score"] + out["m_score"]
    out["rfm_avg"]   = out["rfm_total"] / 3
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. K-Means Clustering
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_FEATURES = [
    "recency_days", "frequency", "monetary",
    "avg_order_value", "discount_usage_rate",
    "orders_per_month", "recent_spend_ratio",
]

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(3, 9)) -> dict:
    """Elbow + silhouette sweep to find optimal k."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return {"k": list(k_range), "inertia": inertias, "silhouette": silhouettes}


def run_kmeans(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Fits K-Means on CLUSTER_FEATURES.
    Adds: cluster_id, segment_label, segment_color columns.
    """
    feats = [f for f in CLUSTER_FEATURES if f in df.columns]
    X = df[feats].fillna(0).values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    df = df.copy()
    df["cluster_id"] = km.fit_predict(X_scaled)

    # ── label each cluster by its mean RFM profile ───────────────────────
    cluster_means = (
        df.groupby("cluster_id")[["recency_days","frequency","monetary","discount_usage_rate"]]
        .mean()
    )
    labels = _label_clusters(cluster_means, n_clusters)
    df["segment_label"] = df["cluster_id"].map(labels)
    df["segment_color"] = df["segment_label"].map(
        lambda s: PALETTE.get(s, "#BDC3C7")
    )
    return df, km, scaler, feats


def _label_clusters(means: pd.DataFrame, k: int) -> dict[int, str]:
    """
    Heuristic cluster labelling based on relative RFM centroid positions.
    Works for any k between 3 and 7.
    """
    # normalise 0-1 within each metric
    normed = means.copy()
    for col in normed.columns:
        rng = normed[col].max() - normed[col].min()
        if rng > 0:
            normed[col] = (normed[col] - normed[col].min()) / rng

    # invert recency (lower days = better)
    normed["recency_days"] = 1 - normed["recency_days"]

    # composite score
    normed["score"] = (
        normed["recency_days"]      * 0.30 +
        normed["frequency"]         * 0.30 +
        normed["monetary"]          * 0.25 +
        (1 - normed["discount_usage_rate"]) * 0.15
    )
    ranked = normed["score"].rank(ascending=False).astype(int)

    label_pool = [
        "Champions",          # rank 1 — best overall
        "Loyal Customers",    # rank 2 — high freq, moderate recency
        "Promising",          # rank 3 — recent but lower freq
        "Discount Dependent", # rank 4 — high discount, moderate value
        "At Risk",            # rank 5 — low recency, was active
        "Lost / Inactive",    # rank 6
        "Need Attention",     # rank 7
    ]
    mapping = {}
    for cluster_id, rank in ranked.items():
        idx = min(rank - 1, len(label_pool) - 1)
        mapping[cluster_id] = label_pool[idx]
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pareto Analysis
# ─────────────────────────────────────────────────────────────────────────────

def pareto_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Returns cumulative revenue contribution sorted by customer value."""
    out = df[["customer_id","monetary"]].copy().sort_values("monetary", ascending=False)
    out["cum_revenue"]       = out["monetary"].cumsum()
    out["cum_revenue_pct"]   = out["cum_revenue"] / out["monetary"].sum() * 100
    out["cum_customer_pct"]  = (np.arange(1, len(out)+1) / len(out)) * 100
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Segment Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_rev = df["monetary"].sum()
    summary = df.groupby("segment_label").agg(
        customer_count   = ("customer_id",        "count"),
        avg_recency_days = ("recency_days",        "mean"),
        avg_orders       = ("frequency",           "mean"),
        avg_revenue      = ("monetary",            "mean"),
        total_revenue    = ("monetary",            "sum"),
        avg_discount_pct = ("discount_usage_rate", "mean"),
    ).reset_index()
    summary["revenue_share_pct"] = (summary["total_revenue"] / total_rev * 100).round(1)
    for col in ["avg_recency_days","avg_orders","avg_revenue","total_revenue","avg_discount_pct"]:
        summary[col] = summary[col].round(1)
    return summary.sort_values("total_revenue", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_segment_distribution(df: pd.DataFrame, save: bool = True):
    seg_counts = df["segment_label"].value_counts()
    colors     = [PALETTE.get(s, "#BDC3C7") for s in seg_counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Customer Segment Distribution", fontsize=15, fontweight="bold", y=1.01)

    # pie
    axes[0].pie(
        seg_counts.values,
        labels=seg_counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
        textprops={"fontsize": 9},
    )
    axes[0].set_title("Customer Count Share", fontsize=12, fontweight="bold")

    # bar — revenue by segment
    rev_by_seg = df.groupby("segment_label")["monetary"].sum().sort_values(ascending=False)
    bar_colors = [PALETTE.get(s, "#BDC3C7") for s in rev_by_seg.index]
    axes[1].barh(rev_by_seg.index, rev_by_seg.values / 1e6, color=bar_colors, edgecolor="white")
    axes[1].set_xlabel("Total Revenue (₹ Million)", fontsize=10)
    _style_ax(axes[1], title="Revenue Contribution by Segment")

    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "01_segment_distribution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '01_segment_distribution.png'}")
    return fig


def plot_rfm_scatter(df: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(10, 7))
    for seg, grp in df.groupby("segment_label"):
        ax.scatter(
            grp["recency_days"],
            grp["frequency"],
            c=PALETTE.get(seg, "#BDC3C7"),
            label=seg,
            alpha=0.55,
            s=grp["monetary"] / grp["monetary"].max() * 120 + 10,
            edgecolors="none",
        )
    _style_ax(ax, "RFM Scatter (size ∝ Revenue)", "Recency (days since last purchase)", "Purchase Frequency")
    ax.legend(title="Segment", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "02_rfm_scatter.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '02_rfm_scatter.png'}")
    return fig


def plot_pareto(df: pd.DataFrame, save: bool = True):
    pareto = pareto_analysis(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pareto["cum_customer_pct"], pareto["cum_revenue_pct"],
            color="#2C3E50", lw=2.5, label="Actual")
    ax.plot([0,100],[0,100], "--", color="#95A5A6", lw=1, label="45° line (uniform)")
    # highlight 20-80 point
    idx_20 = (pareto["cum_customer_pct"] - 20).abs().idxmin()
    rev_at_20 = pareto.loc[idx_20, "cum_revenue_pct"]
    ax.annotate(
        f"Top 20% customers\n→ {rev_at_20:.0f}% revenue",
        xy=(20, rev_at_20), xytext=(30, rev_at_20 - 15),
        arrowprops=dict(arrowstyle="->", color="#E74C3C"),
        fontsize=10, color="#E74C3C", fontweight="bold"
    )
    ax.scatter([20], [rev_at_20], color="#E74C3C", zorder=5, s=80)
    _style_ax(ax, "Pareto Curve — Customer Revenue Concentration",
              "Cumulative % of Customers (sorted by revenue desc)",
              "Cumulative % of Revenue")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "03_pareto_curve.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '03_pareto_curve.png'}")
    return fig


def plot_discount_analysis(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Discount Behaviour Analysis", fontsize=14, fontweight="bold")

    # box: discount rate by segment
    seg_order = df.groupby("segment_label")["discount_usage_rate"].median().sort_values().index.tolist()
    colors    = [PALETTE.get(s, "#BDC3C7") for s in seg_order]
    data_list = [df[df["segment_label"]==s]["discount_usage_rate"].values for s in seg_order]
    bp = axes[0].boxplot(data_list, vert=False, patch_artist=True,
                         medianprops=dict(color="white", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[0].set_yticklabels(seg_order, fontsize=9)
    _style_ax(axes[0], "Discount Usage Rate by Segment", "Discount Usage Rate (%)")

    # scatter: discount rate vs LTV
    sc = axes[1].scatter(
        df["discount_usage_rate"], df["monetary"],
        c=df["cluster_id"], cmap="tab10", alpha=0.35, s=15, edgecolors="none"
    )
    z = np.polyfit(df["discount_usage_rate"].fillna(0), df["monetary"], 1)
    p = np.poly1d(z)
    xline = np.linspace(0, 100, 100)
    axes[1].plot(xline, p(xline), "r--", lw=1.5, label=f"Trend (slope={z[0]:.1f})")
    axes[1].legend(fontsize=9)
    _style_ax(axes[1], "Discount Rate vs Customer Revenue (LTV proxy)",
              "Discount Usage Rate (%)", "Total Revenue (₹)")

    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "04_discount_analysis.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '04_discount_analysis.png'}")
    return fig


def run_segmentation(ml_features: pd.DataFrame, verbose: bool = True):
    """Master function — runs full segmentation pipeline & saves charts."""
    print("🔵  Running customer segmentation...")

    # RFM scoring
    df = compute_rfm_scores(ml_features)

    # K-Means (k=5)
    df, km, scaler, feats = run_kmeans(df, n_clusters=5)

    # summary
    summary = segment_summary(df)
    if verbose:
        print("\n📊  Segment Summary:")
        print(summary.to_string(index=False))

    # charts
    plot_segment_distribution(df)
    plot_rfm_scatter(df)
    plot_pareto(df)
    plot_discount_analysis(df)
    print("  ✅  Segmentation complete.\n")

    return df, summary, km, scaler
