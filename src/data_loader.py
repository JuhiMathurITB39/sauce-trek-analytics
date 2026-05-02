"""
src/data_loader.py
------------------
Loads CSVs into an in-memory SQLite database and executes all SQL scripts
to materialise views. Returns clean DataFrames for downstream analysis.
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SQL_DIR  = BASE_DIR / "sql"


# ── helpers ──────────────────────────────────────────────────────────────────

def _exec_sql_file(conn: sqlite3.Connection, path: Path) -> None:
    """Execute a .sql file, stripping comments before splitting on semicolons."""
    import re
    sql_text = path.read_text()
    # Remove block comments /* ... */
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    # Remove single-line comments -- ...
    sql_text = re.sub(r'--[^\n]*', '', sql_text)
    # Split on semicolons and run each statement
    stmts = [s.strip() for s in sql_text.split(";") if s.strip()]
    for stmt in stmts:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            pass   # silently skip errors (optional/diagnostic statements)
    conn.commit()


def build_database(verbose: bool = True) -> sqlite3.Connection:
    """
    1. Load CSV files into SQLite tables.
    2. Run all SQL scripts to build views.
    Returns an open SQLite connection.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # ── load raw tables ───────────────────────────────────────────────────
    tables = {
        "customers":   DATA_DIR / "customers.csv",
        "orders":      DATA_DIR / "orders.csv",
        "order_items": DATA_DIR / "order_items.csv",
        "products":    DATA_DIR / "products.csv",
    }
    for tbl, path in tables.items():
        if verbose:
            print(f"  Loading {path.name} → {tbl}")
        df = pd.read_csv(path)
        df.to_sql(tbl, conn, if_exists="replace", index=False)

    # ── run SQL scripts in order ──────────────────────────────────────────
    sql_scripts = sorted(SQL_DIR.glob("*.sql"))
    for script in sql_scripts:
        if verbose:
            print(f"  Executing {script.name}")
        _exec_sql_file(conn, script)

    if verbose:
        print("  ✅ Database ready.\n")
    return conn


# ── public API ────────────────────────────────────────────────────────────────

def load_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """
    Returns a dict of DataFrames:
        customers, orders, order_items, products,
        customer_stats, rfm_segments, ml_features,
        cohort_retention, cohort_revenue
    """
    conn = build_database(verbose=verbose)

    datasets = {}

    # raw tables
    for tbl in ("customers", "orders", "order_items", "products"):
        datasets[tbl] = pd.read_sql(f"SELECT * FROM {tbl}", conn)

    # aggregated views
    view_map = {
        "customer_stats":  "v_customer_stats",
        "rfm_segments":    "v_rfm_segments",
        "ml_features":     "v_ml_features",
        "cohort_retention":"v_cohort_retention",
        "cohort_revenue":  "v_cohort_revenue",
        "order_gaps":      "v_customer_purchase_intervals",
    }
    for key, view in view_map.items():
        try:
            datasets[key] = pd.read_sql(f"SELECT * FROM {view}", conn)
        except Exception as e:
            pass  # handled by Python fallback below

    conn.close()

    # ── Python fallback for ml_features ──────────────────────────────────
    if "ml_features" not in datasets or datasets["ml_features"].empty:
        if verbose:
            print("  [info] Computing ml_features via Python (SQL view fallback)...")
        datasets["ml_features"] = _compute_ml_features_python(
            datasets["orders"], datasets["order_items"],
            datasets["products"], datasets["customers"]
        )

    # ── Python fallback for cohort retention ─────────────────────────────
    if "cohort_retention" not in datasets or datasets["cohort_retention"].empty:
        if verbose:
            print("  [info] Computing cohort_retention via Python...")
        datasets["cohort_retention"] = _compute_cohort_python(datasets["orders"])

    return datasets


def _compute_ml_features_python(orders, items, products, customers) -> pd.DataFrame:
    """Full Python re-implementation of SQL v_ml_features view."""
    REF_DATE = pd.Timestamp("2024-06-30")

    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    # Core aggregates
    agg = orders.groupby("customer_id").agg(
        frequency=("order_id", "nunique"),
        monetary=("total_amount", "sum"),
        avg_order_value=("total_amount", "mean"),
        max_order_value=("total_amount", "max"),
        min_order_value=("total_amount", "min"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
        discounted_orders=("discount_applied", lambda x: (x == "Yes").sum()),
        total_disc_pct=("discount_percentage", "sum"),
    ).reset_index()

    agg["recency_days"]    = (REF_DATE - agg["last_order_date"]).dt.days
    agg["lifespan_days"]   = (agg["last_order_date"] - agg["first_order_date"]).dt.days.clip(lower=1)
    agg["discount_usage_rate"] = (agg["discounted_orders"] / agg["frequency"] * 100).round(2)
    agg["avg_discount_pct"]    = (agg["total_disc_pct"] / agg["discounted_orders"].clip(lower=1)).round(2)
    agg["orders_per_month"]    = (agg["frequency"] / (agg["lifespan_days"] / 30.0)).clip(upper=30).round(3)

    # Tenure
    customers = customers.copy()
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    customers["tenure_days"] = (REF_DATE - customers["signup_date"]).dt.days
    agg = agg.merge(customers[["customer_id","city","channel","tenure_days"]], on="customer_id", how="left")
    agg["days_since_signup"] = agg["tenure_days"]

    # Recent spend ratio (last 90 days)
    orders["days_to_ref"] = (REF_DATE - orders["order_date"]).dt.days
    recent = orders[orders["days_to_ref"] <= 90].groupby("customer_id")["total_amount"].sum().rename("rev_last_90")
    agg = agg.merge(recent, on="customer_id", how="left")
    agg["recent_spend_ratio"] = (agg["rev_last_90"].fillna(0) / agg["monetary"].clip(lower=1)).round(3)

    # Category preference
    oi_merged = items.merge(products[["product_id","category"]], on="product_id", how="left")
    oi_merged = oi_merged.merge(orders[["order_id","customer_id"]], on="order_id", how="left")
    oi_merged["line_spend"] = oi_merged["quantity"] * oi_merged["price"]
    cat_spend = oi_merged.groupby(["customer_id","category"])["line_spend"].sum().unstack(fill_value=0)
    cat_spend["total"] = cat_spend.sum(axis=1)
    for cat in ["mayo","sauces","dips"]:
        if cat in cat_spend.columns:
            cat_spend[f"{cat}_spend_share"] = (cat_spend[cat] / cat_spend["total"]).round(3)
        else:
            cat_spend[f"{cat}_spend_share"] = 0.0
    cat_spend["distinct_products_bought"] = oi_merged.groupby("customer_id")["product_id"].nunique()
    agg = agg.merge(
        cat_spend[["mayo_spend_share","sauces_spend_share","dips_spend_share","distinct_products_bought"]],
        on="customer_id", how="left"
    )

    # Target variable
    agg["is_churned"] = (agg["recency_days"] > 60).astype(int)

    # Clean up
    drop_cols = ["first_order_date","last_order_date","discounted_orders","total_disc_pct","rev_last_90"]
    agg = agg.drop(columns=[c for c in drop_cols if c in agg.columns])
    agg = agg.fillna(0)

    return agg


def _compute_cohort_python(orders: pd.DataFrame) -> pd.DataFrame:
    """Python cohort retention computation."""
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["order_ym"]   = orders["order_date"].dt.to_period("M")

    first_order = orders.groupby("customer_id")["order_ym"].min().rename("cohort_month").reset_index()
    orders = orders.merge(first_order, on="customer_id")
    orders["period_number"] = (orders["order_ym"] - orders["cohort_month"]).apply(lambda x: x.n)

    cohort_sizes = first_order.groupby("cohort_month")["customer_id"].nunique().rename("cohort_size")
    retention    = orders[orders["period_number"] >= 0].groupby(
        ["cohort_month","period_number"]
    )["customer_id"].nunique().rename("active_customers").reset_index()

    retention = retention.merge(cohort_sizes, on="cohort_month")
    retention["retention_rate_pct"] = (retention["active_customers"] / retention["cohort_size"] * 100).round(1)
    retention["cohort_month"] = retention["cohort_month"].astype(str)
    return retention[retention["period_number"] <= 18]


def quick_stats(datasets: dict) -> None:
    """Print a summary of loaded datasets."""
    print("\n📦  Dataset Summary")
    print("─" * 40)
    for k, df in datasets.items():
        print(f"  {k:<22}: {len(df):>8,} rows  ×  {df.shape[1]} cols")
    print()
