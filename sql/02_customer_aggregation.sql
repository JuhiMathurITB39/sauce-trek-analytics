-- =============================================================================
-- 02_customer_aggregation.sql
-- Sauce Trek Analytics — Customer-Level KPIs
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1.  Core customer revenue & frequency table
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_customer_stats;
CREATE VIEW v_customer_stats AS
SELECT
    o.customer_id,
    c.city,
    c.signup_date,
    c.channel,

    -- Revenue
    ROUND(SUM(o.total_amount), 2)                              AS total_revenue,
    ROUND(AVG(o.total_amount), 2)                              AS avg_order_value,
    ROUND(MAX(o.total_amount), 2)                              AS max_order_value,

    -- Frequency
    COUNT(DISTINCT o.order_id)                                 AS order_count,

    -- Time window
    MIN(o.order_date)                                          AS first_order_date,
    MAX(o.order_date)                                          AS last_order_date,

    -- Active lifespan in days (between first and last purchase)
    CAST(
        JULIANDAY(MAX(o.order_date)) -
        JULIANDAY(MIN(o.order_date))
    AS INTEGER)                                                AS lifespan_days,

    -- Days since last purchase (recency proxy; ref date = 2024-06-30)
    CAST(
        JULIANDAY('2024-06-30') -
        JULIANDAY(MAX(o.order_date))
    AS INTEGER)                                                AS days_since_last_purchase,

    -- Discount behaviour
    SUM(CASE WHEN o.discount_applied = 'Yes' THEN 1 ELSE 0 END) AS discounted_orders,
    ROUND(
        100.0 * SUM(CASE WHEN o.discount_applied = 'Yes' THEN 1 ELSE 0 END)
              / COUNT(DISTINCT o.order_id),
    2)                                                         AS discount_usage_rate_pct,
    ROUND(AVG(
        CASE WHEN o.discount_applied = 'Yes'
             THEN o.discount_percentage
        END
    ), 2)                                                      AS avg_discount_pct

FROM   v_orders_clean o
JOIN   customers      c ON o.customer_id = c.customer_id
GROUP  BY o.customer_id, c.city, c.signup_date, c.channel;


-- ---------------------------------------------------------------------------
-- 2.  Purchase-interval statistics (inter-order gap analysis)
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_order_gaps;
CREATE VIEW v_order_gaps AS
WITH ranked AS (
    SELECT
        customer_id,
        order_date,
        LAG(order_date) OVER (
            PARTITION BY customer_id ORDER BY order_date
        ) AS prev_order_date
    FROM v_orders_clean
)
SELECT
    customer_id,
    order_date,
    CAST(JULIANDAY(order_date) - JULIANDAY(prev_order_date) AS INTEGER) AS days_gap
FROM ranked
WHERE prev_order_date IS NOT NULL;

DROP VIEW IF EXISTS v_customer_purchase_intervals;
CREATE VIEW v_customer_purchase_intervals AS
SELECT
    customer_id,
    ROUND(AVG(days_gap), 1)  AS avg_purchase_interval_days,
    ROUND(MIN(days_gap), 1)  AS min_purchase_interval_days,
    ROUND(MAX(days_gap), 1)  AS max_purchase_interval_days
FROM   v_order_gaps
GROUP  BY customer_id;


-- ---------------------------------------------------------------------------
-- 3.  Product-mix per customer (what do they buy?)
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_customer_product_mix;
CREATE VIEW v_customer_product_mix AS
SELECT
    o.customer_id,
    p.category,
    SUM(oi.quantity)                                    AS total_units,
    ROUND(SUM(oi.quantity * oi.price), 2)               AS category_revenue,
    COUNT(DISTINCT p.product_id)                        AS distinct_products
FROM   v_items_clean   oi
JOIN   v_orders_clean  o  ON oi.order_id   = o.order_id
JOIN   products        p  ON oi.product_id = p.product_id
GROUP  BY o.customer_id, p.category;


-- ---------------------------------------------------------------------------
-- 4.  Top 20% revenue validation (Pareto check)
-- ---------------------------------------------------------------------------
-- Run this query to see how much revenue the top quintile drives:
/*
WITH ranked AS (
    SELECT
        customer_id,
        total_revenue,
        NTILE(5) OVER (ORDER BY total_revenue DESC) AS quintile
    FROM v_customer_stats
)
SELECT
    quintile,
    COUNT(*)                        AS customer_count,
    ROUND(SUM(total_revenue), 0)    AS quintile_revenue,
    ROUND(
        100.0 * SUM(total_revenue) /
        SUM(SUM(total_revenue)) OVER (),
    1)                              AS revenue_share_pct
FROM   ranked
GROUP  BY quintile
ORDER  BY quintile;
*/
