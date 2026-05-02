"""
generate_data.py
----------------
Generates a realistic synthetic FMCG dataset for Sauce Trek analytics.

Outputs (saved to data/):
  customers.csv   - 5,000 customers
  orders.csv      - 50,000+ orders
  order_items.csv - ~100,000 line items
  products.csv    - 24 SKUs across mayo / sauces / dips
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random, os

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__))

# ── constants ────────────────────────────────────────────────────────────────
N_CUSTOMERS   = 5_000
SIM_START     = datetime(2022, 1, 1)
SIM_END       = datetime(2024, 6, 30)
SIM_DAYS      = (SIM_END - SIM_START).days     # 910 days

CITIES = [
    ("Mumbai",    0.20), ("Delhi",     0.18), ("Bangalore",  0.14),
    ("Hyderabad", 0.10), ("Chennai",   0.09), ("Indore",     0.07),
    ("Pune",      0.06), ("Kolkata",   0.06), ("Jaipur",     0.05),
    ("Ahmedabad", 0.05),
]
CITY_NAMES, CITY_PROBS = zip(*CITIES)

# Customer behavioural archetypes (drives realistic distributions)
ARCHETYPES = {
    "high_value":          {"weight": 0.10, "order_rate": 0.06, "aov_mean": 850,  "aov_std": 200,  "discount_prob": 0.20, "churn_prob": 0.05},
    "loyal":               {"weight": 0.20, "order_rate": 0.04, "aov_mean": 550,  "aov_std": 150,  "discount_prob": 0.25, "churn_prob": 0.10},
    "promising":           {"weight": 0.15, "order_rate": 0.025,"aov_mean": 420,  "aov_std": 120,  "discount_prob": 0.30, "churn_prob": 0.20},
    "discount_dependent":  {"weight": 0.15, "order_rate": 0.03, "aov_mean": 380,  "aov_std": 100,  "discount_prob": 0.75, "churn_prob": 0.30},
    "at_risk":             {"weight": 0.15, "order_rate": 0.015,"aov_mean": 300,  "aov_std": 100,  "discount_prob": 0.40, "churn_prob": 0.50},
    "occasional":          {"weight": 0.15, "order_rate": 0.008,"aov_mean": 280,  "aov_std": 90,   "discount_prob": 0.35, "churn_prob": 0.60},
    "one_time":            {"weight": 0.10, "order_rate": 0.002,"aov_mean": 250,  "aov_std": 80,   "discount_prob": 0.45, "churn_prob": 0.90},
}

# ── 1. Products ──────────────────────────────────────────────────────────────
def build_products() -> pd.DataFrame:
    rows = [
        # (product_id, category, product_name, unit_price)
        ("P001", "mayo",   "Classic Eggless Mayo 200g",      89),
        ("P002", "mayo",   "Classic Eggless Mayo 400g",     159),
        ("P003", "mayo",   "Garlic Herb Mayo 200g",          99),
        ("P004", "mayo",   "Spicy Chipotle Mayo 200g",       99),
        ("P005", "mayo",   "Avocado Mayo 200g",             119),
        ("P006", "mayo",   "Classic Eggless Mayo 1kg",      349),
        ("P007", "mayo",   "Vegan Olive Mayo 200g",         129),
        ("P008", "mayo",   "Sriracha Mayo 200g",             99),
        ("P009", "sauces", "Tangy Tomato Ketchup 500g",      79),
        ("P010", "sauces", "Smoky BBQ Sauce 300g",           99),
        ("P011", "sauces", "Hot Chilli Sauce 200g",          89),
        ("P012", "sauces", "Schezwan Stir-Fry Sauce 250g",  109),
        ("P013", "sauces", "Pasta Arrabbiata Sauce 350g",   139),
        ("P014", "sauces", "Teriyaki Glaze 200g",           129),
        ("P015", "sauces", "Mustard Sauce 200g",             79),
        ("P016", "sauces", "Green Chutney Sauce 200g",       89),
        ("P017", "dips",   "Hummus Classic 200g",           149),
        ("P018", "dips",   "Hummus Roasted Red Pepper 200g",159),
        ("P019", "dips",   "Tzatziki Dip 200g",             169),
        ("P020", "dips",   "Guacamole 200g",                179),
        ("P021", "dips",   "Spinach Artichoke Dip 200g",    149),
        ("P022", "dips",   "Baba Ganoush 200g",             149),
        ("P023", "dips",   "Cheese & Herb Dip 200g",        159),
        ("P024", "dips",   "Sour Cream & Onion Dip 200g",  139),
    ]
    df = pd.DataFrame(rows, columns=["product_id", "category", "product_name", "unit_price"])
    df["margin_pct"] = np.where(
        df["category"] == "mayo", 0.38,
        np.where(df["category"] == "sauces", 0.32, 0.30)
    )
    return df

# ── 2. Customers ─────────────────────────────────────────────────────────────
def build_customers() -> pd.DataFrame:
    archetype_keys = list(ARCHETYPES.keys())
    archetype_wts  = [ARCHETYPES[k]["weight"] for k in archetype_keys]

    cust_id  = [f"C{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]
    city     = np.random.choice(CITY_NAMES, size=N_CUSTOMERS, p=CITY_PROBS)
    archetype= np.random.choice(archetype_keys, size=N_CUSTOMERS, p=archetype_wts)

    # signup spread across simulation window (earlier customers have more history)
    signup_offset = np.random.randint(0, SIM_DAYS - 90, size=N_CUSTOMERS)
    signup_date   = [SIM_START + timedelta(days=int(d)) for d in signup_offset]

    ages     = np.random.randint(22, 55, size=N_CUSTOMERS)
    channels = np.random.choice(
        ["instagram", "referral", "website", "modern_trade", "distributor"],
        size=N_CUSTOMERS,
        p=[0.30, 0.20, 0.25, 0.15, 0.10]
    )

    return pd.DataFrame({
        "customer_id":   cust_id,
        "city":          city,
        "signup_date":   signup_date,
        "age":           ages,
        "channel":       channels,
        "archetype":     archetype,          # kept for simulation; drop for "clean" dataset
    })

# ── 3. Orders + Order Items ──────────────────────────────────────────────────
def build_orders_and_items(customers: pd.DataFrame, products: pd.DataFrame):
    order_rows = []
    item_rows  = []
    order_id   = 1

    product_ids = products["product_id"].tolist()
    # category weights for basket building
    cat_weights = {"mayo": 0.50, "sauces": 0.30, "dips": 0.20}
    prod_by_cat = products.groupby("category")["product_id"].apply(list).to_dict()

    prices = products.set_index("product_id")["unit_price"].to_dict()

    for _, cust in customers.iterrows():
        arch     = ARCHETYPES[cust["archetype"]]
        signup   = cust["signup_date"]
        cust_days= (SIM_END - signup).days
        if cust_days <= 0:
            continue

        # determine if customer has already churned
        churned      = np.random.random() < arch["churn_prob"]
        if churned:
            # churn happens in first 30-50% of their history
            active_till = int(cust_days * np.random.uniform(0.10, 0.50))
        else:
            active_till = cust_days

        # expected number of orders in active window
        expected_orders = max(1, int(active_till * arch["order_rate"]))
        n_orders = max(1, np.random.poisson(expected_orders))

        if n_orders == 0:
            continue

        # sample order dates uniformly in active window
        if active_till <= 1:
            offsets = [0]
        else:
            offsets = sorted(np.random.choice(range(active_till), size=n_orders, replace=False))

        for off in offsets:
            o_date          = signup + timedelta(days=int(off))
            discount_applied= np.random.random() < arch["discount_prob"]
            disc_pct        = 0
            if discount_applied:
                disc_pct = np.random.choice([5, 10, 15, 20, 25], p=[0.20, 0.30, 0.25, 0.15, 0.10])

            # basket: 1-5 items, category-weighted
            n_items = np.random.choice([1, 2, 3, 4, 5], p=[0.35, 0.30, 0.20, 0.10, 0.05])
            basket_amount = 0.0
            order_id_str  = f"O{str(order_id).zfill(7)}"

            for _ in range(n_items):
                cat = np.random.choice(
                    list(cat_weights.keys()), p=list(cat_weights.values())
                )
                pid = np.random.choice(prod_by_cat[cat])
                qty = np.random.choice([1, 2, 3], p=[0.65, 0.25, 0.10])
                unit_p = prices[pid]
                line_total = qty * unit_p
                basket_amount += line_total
                item_rows.append({
                    "order_id":   order_id_str,
                    "product_id": pid,
                    "quantity":   qty,
                    "price":      unit_p,
                    "line_total": round(line_total, 2),
                })

            # apply discount
            total_after_disc = round(basket_amount * (1 - disc_pct / 100), 2)

            # small noise on AOV
            aov_noise = np.random.normal(0, arch["aov_std"] * 0.05)
            total_after_disc = max(49.0, total_after_disc + aov_noise)

            order_rows.append({
                "order_id":          order_id_str,
                "customer_id":       cust["customer_id"],
                "order_date":        o_date.date(),
                "total_amount":      round(total_after_disc, 2),
                "discount_applied":  "Yes" if discount_applied else "No",
                "discount_percentage": disc_pct,
            })
            order_id += 1

    orders = pd.DataFrame(order_rows)
    items  = pd.DataFrame(item_rows)
    return orders, items

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("⚙️  Building products...")
    products  = build_products()

    print("⚙️  Building customers...")
    customers = build_customers()

    print("⚙️  Simulating orders & items (this takes ~30 seconds)...")
    orders, items = build_orders_and_items(customers, products)

    # drop internal archetype column before saving "clean" customer table
    customers_clean = customers.drop(columns=["archetype"])

    # save
    products.to_csv(os.path.join(OUT_DIR, "products.csv"), index=False)
    customers_clean.to_csv(os.path.join(OUT_DIR, "customers.csv"), index=False)
    orders.to_csv(os.path.join(OUT_DIR, "orders.csv"), index=False)
    items.to_csv(os.path.join(OUT_DIR, "order_items.csv"), index=False)

    print("\n✅  Dataset generated:")
    print(f"   Products  : {len(products):>7,}")
    print(f"   Customers : {len(customers_clean):>7,}")
    print(f"   Orders    : {len(orders):>7,}")
    print(f"   Order Items: {len(items):>6,}")
    print(f"\n   Saved to → {OUT_DIR}")

if __name__ == "__main__":
    main()
