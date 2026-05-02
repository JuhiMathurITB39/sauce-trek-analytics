-- =============================================================================
-- 04_cohort_analysis.sql
-- Sauce Trek Analytics — Monthly Cohort Retention
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1.  Assign each customer to their acquisition cohort (first-order month)
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_cohort_base;
CREATE VIEW v_cohort_base AS
SELECT
    customer_id,
    -- SQLite: substr(date, 1, 7) gives 'YYYY-MM'
    -- Postgres: TO_CHAR(MIN(order_date), 'YYYY-MM')
    SUBSTR(CAST(MIN(order_date) AS TEXT), 1, 7) AS cohort_month
FROM   v_orders_clean
GROUP  BY customer_id;


-- ---------------------------------------------------------------------------
-- 2.  For each order, calculate how many months after cohort it occurred
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_cohort_activity;
CREATE VIEW v_cohort_activity AS
SELECT
    b.customer_id,
    b.cohort_month,
    SUBSTR(CAST(o.order_date AS TEXT), 1, 7) AS order_month,

    -- Month offset: number of months from first purchase month
    -- SQLite workaround (no DATEDIFF months): use year*12 + month arithmetic
    (
        (CAST(SUBSTR(CAST(o.order_date AS TEXT), 1, 4) AS INTEGER) * 12
         + CAST(SUBSTR(CAST(o.order_date AS TEXT), 6, 2) AS INTEGER))
        -
        (CAST(SUBSTR(b.cohort_month, 1, 4) AS INTEGER) * 12
         + CAST(SUBSTR(b.cohort_month, 6, 2) AS INTEGER))
    ) AS period_number          -- 0 = acquisition month, 1 = month 1, etc.

FROM   v_cohort_base  b
JOIN   v_orders_clean o ON b.customer_id = o.customer_id;


-- ---------------------------------------------------------------------------
-- 3.  Cohort size (number of customers in each acquisition cohort)
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_cohort_sizes;
CREATE VIEW v_cohort_sizes AS
SELECT
    cohort_month,
    COUNT(DISTINCT customer_id) AS cohort_size
FROM   v_cohort_base
GROUP  BY cohort_month;


-- ---------------------------------------------------------------------------
-- 4.  Retention table: unique active customers per cohort × period
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_cohort_retention;
CREATE VIEW v_cohort_retention AS
SELECT
    a.cohort_month,
    a.period_number,
    COUNT(DISTINCT a.customer_id)            AS active_customers,
    s.cohort_size,
    ROUND(
        100.0 * COUNT(DISTINCT a.customer_id) / s.cohort_size,
    1)                                       AS retention_rate_pct
FROM   v_cohort_activity a
JOIN   v_cohort_sizes    s ON a.cohort_month = s.cohort_month
WHERE  a.period_number  >= 0
  AND  a.period_number  <= 18      -- cap at 18 months
GROUP  BY a.cohort_month, a.period_number, s.cohort_size
ORDER  BY a.cohort_month, a.period_number;


-- ---------------------------------------------------------------------------
-- 5.  Average retention curve (across all cohorts) for each period
-- ---------------------------------------------------------------------------
/*
SELECT
    period_number,
    ROUND(AVG(retention_rate_pct), 1) AS avg_retention_pct,
    COUNT(DISTINCT cohort_month)       AS cohorts_included
FROM   v_cohort_retention
GROUP  BY period_number
ORDER  BY period_number;
*/


-- ---------------------------------------------------------------------------
-- 6.  Revenue cohort table (average revenue per customer per month)
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_cohort_revenue;
CREATE VIEW v_cohort_revenue AS
SELECT
    b.cohort_month,
    (
        (CAST(SUBSTR(CAST(o.order_date AS TEXT), 1, 4) AS INTEGER) * 12
         + CAST(SUBSTR(CAST(o.order_date AS TEXT), 6, 2) AS INTEGER))
        -
        (CAST(SUBSTR(b.cohort_month, 1, 4) AS INTEGER) * 12
         + CAST(SUBSTR(b.cohort_month, 6, 2) AS INTEGER))
    )                                            AS period_number,
    s.cohort_size,
    ROUND(SUM(o.total_amount) / s.cohort_size, 2) AS avg_revenue_per_customer
FROM   v_cohort_base  b
JOIN   v_orders_clean o ON b.customer_id = o.customer_id
JOIN   v_cohort_sizes s ON b.cohort_month = s.cohort_month
GROUP  BY b.cohort_month, period_number, s.cohort_size
HAVING period_number >= 0 AND period_number <= 18
ORDER  BY b.cohort_month, period_number;
