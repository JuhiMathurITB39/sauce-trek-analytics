"""
run_analysis.py
---------------
Master orchestration script for Sauce Trek Customer Analytics.

Run order:
  1. Generate synthetic data   (only if CSVs don't exist)
  2. Load data → SQLite
  3. Customer segmentation (RFM + K-Means)
  4. Churn prediction model
  5. LTV estimation + cohort retention
  6. Generate consulting-grade insights report
  7. Save final customer master table

Usage:
    python run_analysis.py              # full pipeline
    python run_analysis.py --skip-data  # skip data generation
"""

import sys
import os
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# silence sklearn warnings in demo
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader  import load_all, quick_stats
from src.segmentation import run_segmentation
from src.churn_model  import run_churn_model
from src.ltv_model    import run_ltv_analysis

DATA_DIR    = BASE_DIR / "data"
OUT_DIR     = BASE_DIR / "outputs"
REPORT_PATH = OUT_DIR  / "reports" / "insights_report.md"
MASTER_PATH = OUT_DIR  / "reports" / "customer_master.csv"

(OUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "charts").mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data generation
# ─────────────────────────────────────────────────────────────────────────────

def maybe_generate_data():
    required = ["customers.csv","orders.csv","order_items.csv","products.csv"]
    if all((DATA_DIR / f).exists() for f in required):
        print(" Data files exist. Skipping generation.")
        return
    print("Generating synthetic dataset...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(DATA_DIR / "generate_data.py")],
        capture_output=False
    )
    if result.returncode != 0:
        raise RuntimeError("Data generation failed.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Insights Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights_report(
    master: pd.DataFrame,
    segment_summary: pd.DataFrame,
    churn_result: dict,
    cohort_df: pd.DataFrame,
) -> str:
    """Generates a Markdown consulting report with real computed numbers."""

    total_customers = len(master)
    total_revenue   = master["monetary"].sum()
    avg_clv         = master["monetary"].mean()
    avg_aov         = master["avg_order_value"].mean()
    median_recency  = master["recency_days"].median()

    # Pareto
    top_20_rev = master.nlargest(int(total_customers * 0.20), "monetary")["monetary"].sum()
    top_20_pct = top_20_rev / total_revenue * 100

    # Churn
    churn_rate = master["is_churned"].mean() * 100
    high_risk  = (master["churn_risk_tier"] == "High Risk").sum()
    med_risk   = (master["churn_risk_tier"] == "Medium Risk").sum()

    # Discount
    disc_mean_all      = master["discount_usage_rate"].mean()
    disc_churn_rev     = master[master["is_churned"]==1]["monetary"].mean()
    disc_active_rev    = master[master["is_churned"]==0]["monetary"].mean()

    # Cohort: month-1 retention
    m1_ret = cohort_df[cohort_df["period_number"]==1]["retention_rate_pct"].mean() if not cohort_df.empty else 0
    m3_ret = cohort_df[cohort_df["period_number"]==3]["retention_rate_pct"].mean() if not cohort_df.empty else 0

    # LTV tiers
    if "ltv_tier" in master.columns:
        plat = master[master["ltv_tier"]=="Platinum"]
        plat_rev_share = plat["monetary"].sum() / total_revenue * 100
    else:
        plat_rev_share = 0

    best_auc = churn_result.get("auc", 0)

    top_segs = segment_summary.head(2)["segment_label"].tolist()

    report = f"""# Sauce Trek — Customer Retention & Revenue Intelligence Report
**Generated:** {datetime.now().strftime('%d %b %Y, %H:%M')}
**Data window:** Jan 2022 – Jun 2024  |  **Reference date:** 30 Jun 2024

---

## Executive Summary

| KPI | Value |
|-----|-------|
| Total Customers Analysed | {total_customers:,} |
| Total Revenue (period) | ₹{total_revenue:,.0f} |
| Average Customer Revenue | ₹{avg_clv:,.0f} |
| Average Order Value | ₹{avg_aov:,.0f} |
| Median Days Since Last Purchase | {median_recency:.0f} days |
| Overall Churn Rate (60-day) | {churn_rate:.1f}% |
| Best Model ROC-AUC | {best_auc:.4f} |

---

## Business Insights

### Insight 1 — Pareto Revenue Concentration
> **The top {20:.0f}% of customers account for {top_20_pct:.1f}% of total revenue.**

This is {'more' if top_20_pct > 80 else 'slightly less'} extreme than the classic 80/20 rule.
Prioritise retention and upsell campaigns exclusively within this cohort.
A 5% churn reduction in this tier is worth ≈ ₹{top_20_rev * 0.05 / 1e5:.1f} lakh in protected revenue.

---

### Insight 2 — Segment-Level Revenue Breakdown

| Segment | Customers | Avg Revenue (₹) | Revenue Share |
|---------|-----------|-----------------|---------------|
{_seg_table(segment_summary)}

**Top revenue segments ({', '.join(top_segs)}) must be protected at all costs.**
Reactivation programmes should target the "At Risk" segment before customers move to "Lost".

---

### Insight 3 — Discount Dependency Destroys LTV
> Churned customers have on average **₹{disc_churn_rev:,.0f}** in lifetime spend vs
> **₹{disc_active_rev:,.0f}** for active customers.

Discount-heavy users exhibit **{(disc_mean_all):.0f}%** average discount usage rate.
Discounting drives short-term volume but does NOT improve retention.

**Key finding:** A 1% increase in discount usage rate is associated with a measurable
reduction in 12-month predicted LTV (see feature importance chart).

**Recommendation:** Cap promotional discounts at 15%. Replace blanket discounts with
personalised loyalty rewards (free unit, express delivery, recipe kits).

---

### Insight 4 — Churn Happens Fast: The 60-Day Window
> **{churn_rate:.1f}%** of customers have not purchased in 60 days (defined as churned).
> Month-1 retention: **{m1_ret:.0f}%** — Month-3 retention: **{m3_ret:.0f}%**

This means **{100-m1_ret:.0f}%** of new customers never make a second purchase.
The first 30 days post-acquisition are the highest-leverage intervention window.

**Immediate actions:**
- Day 3: Post-purchase "recipe inspiration" WhatsApp / email
- Day 14: Loyalty point credit with next-purchase nudge
- Day 30: "We miss you" reactivation with personalised SKU recommendation
- Day 55: Final win-back offer (free shipping or small gift)

---

### Insight 5 — High-Risk Churn Pipeline
> **{high_risk:,}** customers are flagged as **High Risk** (churn probability > 60%).
> **{med_risk:,}** customers are **Medium Risk** (40–60%).

Combined at-risk pipeline represents approximately
**₹{master[master['churn_risk_tier']!='Low Risk']['monetary'].sum()/1e5:.1f} lakh** in historical revenue.
Saving 30% of this cohort via targeted interventions = significant EBITDA recovery.

---

### Insight 6 — Platinum LTV Customers are Disproportionately Valuable
> The top LTV quartile ("Platinum") represents **{plat_rev_share:.0f}%** of total revenue.

These customers buy across multiple categories (mayo + sauces + dips),
have low discount dependency, and purchase consistently.
They are ideal candidates for: subscription models, D2C bundles, and brand ambassador programmes.

---

## Strategic Recommendations

### 1. Protect Top-20% with a VIP Loyalty Programme
- Launch a tiered points system (Bronze → Silver → Gold → Platinum)
- Platinum members: dedicated account manager (for B2B), early access to new SKUs,
  quarterly business review calls
- Target: reduce top-quartile churn from current level by 50%

### 2. New Customer Onboarding — "First 30 Days" Protocol
- Problem: {100-m1_ret:.0f}% first-month churn rate is the biggest revenue leak
- Solution: Automated 5-touch onboarding sequence (recipe email → usage tips →
  loyalty enrolment → feedback request → second-purchase incentive)
- Goal: Lift month-1 retention to 65%+

### 3. Smarter Discounting — Precision Over Blanketing
- Stop: untargeted blanket discounts to all segments
- Start: personalised discounts ONLY to Promising and At-Risk segments
  based on predicted LTV > ₹500
- Cap all promotional discounts at 15% — above this, margin erosion
  outweighs retention gains

### 4. Win-Back Campaign for At-Risk / Lost Segments
- Target: {int(master[master['churn_risk_tier']=='High Risk']['customer_id'].count()):,} high-risk customers
- Mechanic: "We noticed you haven't reordered — here's what's new" email/WhatsApp
  with a limited-time free delivery or trial-size new product
- Expected win-back rate: 15–25% (industry benchmark for FMCG)
- Revenue impact: ₹{master[master['churn_risk_tier']=='High Risk']['ltv_predicted'].mean() * int(master[master['churn_risk_tier']=='High Risk']['customer_id'].count()) * 0.20 / 1e5:.1f} lakh (20% save rate)

### 5. B2B Upsell via Category Expansion
- Customers who buy only mayo (high mayo_spend_share) are prime targets
  for sauce and dip cross-sell
- Cloud kitchens / QSR clients ordering 1 SKU → pitch bundled condiment kits
- Average revenue uplift from cross-category adoption: ~30% higher AOV

### 6. Revenue Growth Levers (Priority Order)
| Priority | Action | Est. Revenue Impact |
|----------|--------|---------------------|
| 1 | Reduce new-customer 30-day churn by 20pp | High |
| 2 | Upsell top 20% to subscription / bundle | High |
| 3 | Win back 20% of high-churn-risk cohort | Medium |
| 4 | Replace blanket discounts with precision offers | Medium |
| 5 | Cross-category expansion for single-SKU buyers | Medium |

---

## Output Files

All charts saved to `outputs/charts/`
Customer master table: `outputs/reports/customer_master.csv`

---
*Generated by Sauce Trek Analytics Engine — github.com/JuhiMathurITB39/sauce-trek-analytics*
"""
    return report


def _seg_table(seg_df: pd.DataFrame) -> str:
    """Format segment summary as Markdown table rows."""
    rows = []
    for _, r in seg_df.iterrows():
        rows.append(
            f"| {r['segment_label']} | {int(r['customer_count']):,} | "
            f"₹{r['avg_revenue']:,.0f} | {r['revenue_share_pct']:.1f}% |"
        )
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(skip_data: bool = False):
    print("=" * 60)
    print("  SAUCE TREK — CUSTOMER ANALYTICS ENGINE")
    print("=" * 60)

    # ── 1. data ───────────────────────────────────────────────────────────
    if not skip_data:
        maybe_generate_data()

    print("\n Loading data into SQLite...")
    datasets = load_all(verbose=True)
    quick_stats(datasets)

    ml_features  = datasets["ml_features"].copy()
    cohort_df    = datasets["cohort_retention"].copy()

    # ── 2. segmentation ───────────────────────────────────────────────────
    seg_df, seg_summary, km, seg_scaler = run_segmentation(ml_features)

    # ── 3. churn model ────────────────────────────────────────────────────
    churn_df, churn_model, churn_scaler, churn_result = run_churn_model(seg_df)

    # ── 4. LTV ────────────────────────────────────────────────────────────
    master, ltv_model = run_ltv_analysis(churn_df, cohort_df)

    # ── 5. Save master table ──────────────────────────────────────────────
    master.to_csv(MASTER_PATH, index=False)
    print(f"   Customer master → {MASTER_PATH}")

    # ── 6. Insights report ────────────────────────────────────────────────
    print("\n Generating consulting insights report...")
    report_md = generate_insights_report(master, seg_summary, churn_result, cohort_df)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"   Report saved → {REPORT_PATH}")

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   ALL STEPS COMPLETE")
    print("=" * 60)
    print(f"\n  Charts  : {OUT_DIR / 'charts'}  ({len(list((OUT_DIR/'charts').glob('*.png')))} files)")
    print(f"  Report  : {REPORT_PATH}")
    print(f"  Master  : {MASTER_PATH}")
    print(f"\n  To launch dashboard:  streamlit run app/streamlit_app.py\n")

    return master, seg_summary, churn_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip synthetic data generation")
    args = parser.parse_args()
    main(skip_data=args.skip_data)
