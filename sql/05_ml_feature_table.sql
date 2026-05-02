-- =============================================================================
-- 05_ml_feature_table.sql
-- Sauce Trek Analytics — Feature Engineering for ML Models
-- Churn definition: no purchase in last 60 days (relative to 2024-06-30)
-- =============================================================================

DROP VIEW IF EXISTS v_ml_features;
CREATE VIEW v_ml_features AS
WITH
-- A. Base order aggregates
base AS (
    SELECT
        o.customer_id,
        COUNT(DISTINCT o.order_id)                           AS frequency,
        ROUND(SUM(o.total_amount), 2)                        AS monetary,
        ROUND(AVG(o.total_amount), 2)                        AS avg_order_value,
        CAST(
            JULIANDAY('2024-06-30') - JULIANDAY(MAX(o.order_date))
        AS INTEGER)                                          AS recency_days,
        CAST(
            JULIANDAY(MAX(o.order_date)) - JULIANDAY(MIN(o.order_date))
        AS INTEGER)                                          AS lifespan_days,
        CAST(
            JULIANDAY('2024-06-30') - JULIANDAY(MIN(o.order_date))
        AS INTEGER)                                          AS days_since_signup,
        ROUND(
            100.0 * SUM(CASE WHEN o.discount_applied='Yes' THEN 1 ELSE 0 END)
                  / COUNT(DISTINCT o.order_id),
        2)                                                   AS discount_usage_rate,
        ROUND(AVG(
            CASE WHEN o.discount_applied='Yes' THEN o.discount_percentage END
        ), 2)                                                AS avg_discount_pct,
        ROUND(MAX(o.total_amount), 2)                        AS max_order_value,
        ROUND(MIN(o.total_amount), 2)                        AS min_order_value

    FROM v_orders_clean o
    GROUP BY o.customer_id
),

-- B. Purchase velocity: orders per active month
velocity AS (
    SELECT
        customer_id,
        ROUND(
            CASE WHEN lifespan_days >= 30
                 THEN frequency / (lifespan_days / 30.0)
                 ELSE frequency
            END,
        3) AS orders_per_month
    FROM base
),

-- C. Category preference (fraction of spend in mayo category)
cat_pref AS (
    SELECT
        o.customer_id,
        ROUND(
            SUM(CASE WHEN p.category = 'mayo'   THEN oi.quantity * oi.price ELSE 0 END)
            / NULLIF(SUM(oi.quantity * oi.price), 0),
        3) AS mayo_spend_share,
        ROUND(
            SUM(CASE WHEN p.category = 'sauces' THEN oi.quantity * oi.price ELSE 0 END)
            / NULLIF(SUM(oi.quantity * oi.price), 0),
        3) AS sauces_spend_share,
        ROUND(
            SUM(CASE WHEN p.category = 'dips'   THEN oi.quantity * oi.price ELSE 0 END)
            / NULLIF(SUM(oi.quantity * oi.price), 0),
        3) AS dips_spend_share,
        COUNT(DISTINCT p.product_id)  AS distinct_products_bought
    FROM v_items_clean oi
    JOIN v_orders_clean o ON oi.order_id  = o.order_id
    JOIN products       p ON oi.product_id = p.product_id
    GROUP BY o.customer_id
),

-- D. Recent vs early spend ratio (last 90 days vs first 90 days)
spend_trend AS (
    SELECT
        customer_id,
        ROUND(SUM(CASE
            WHEN JULIANDAY('2024-06-30') - JULIANDAY(order_date) <= 90
            THEN total_amount ELSE 0
        END), 2) AS revenue_last_90d,
        ROUND(SUM(total_amount), 2) AS revenue_total
    FROM v_orders_clean
    GROUP BY customer_id
),

-- E. Customer metadata
meta AS (
    SELECT customer_id, city, channel,
           CAST(JULIANDAY('2024-06-30') - JULIANDAY(signup_date) AS INTEGER) AS tenure_days
    FROM   customers
)

-- Final feature join
SELECT
    b.customer_id,
    m.city,
    m.channel,
    m.tenure_days,

    -- Core RFM
    b.recency_days,
    b.frequency,
    b.monetary,
    b.avg_order_value,

    -- Extended features
    b.lifespan_days,
    b.days_since_signup,
    b.discount_usage_rate,
    COALESCE(b.avg_discount_pct, 0)          AS avg_discount_pct,
    b.max_order_value,
    b.min_order_value,

    -- Velocity
    v.orders_per_month,

    -- Category preference
    COALESCE(cp.mayo_spend_share,   0)       AS mayo_spend_share,
    COALESCE(cp.sauces_spend_share, 0)       AS sauces_spend_share,
    COALESCE(cp.dips_spend_share,   0)       AS dips_spend_share,
    COALESCE(cp.distinct_products_bought, 0) AS distinct_products_bought,

    -- Spend trend ratio (recent revenue / total; higher = still active)
    COALESCE(
        ROUND(st.revenue_last_90d / NULLIF(st.revenue_total, 0), 3),
    0)                                       AS recent_spend_ratio,

    -- Target variable: churned = 1 if no order in last 60 days
    CASE WHEN b.recency_days > 60 THEN 1 ELSE 0 END AS is_churned

FROM   base       b
JOIN   velocity   v  ON b.customer_id = v.customer_id
LEFT   JOIN cat_pref cp ON b.customer_id = cp.customer_id
LEFT   JOIN spend_trend st ON b.customer_id = st.customer_id
LEFT   JOIN meta  m  ON b.customer_id = m.customer_id;
