-- =============================================================================
-- 03_rfm_features.sql
-- Sauce Trek Analytics — RFM Scoring & Feature Engineering
-- Reference date: 2024-06-30 (last day of dataset)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1.  Raw RFM values per customer
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_rfm_raw;
CREATE VIEW v_rfm_raw AS
SELECT
    customer_id,
    -- Recency: days since last purchase (lower = better)
    CAST(
        JULIANDAY('2024-06-30') - JULIANDAY(MAX(order_date))
    AS INTEGER)                                      AS recency_days,

    -- Frequency: number of distinct orders
    COUNT(DISTINCT order_id)                         AS frequency,

    -- Monetary: total spend
    ROUND(SUM(total_amount), 2)                      AS monetary
FROM   v_orders_clean
GROUP  BY customer_id;


-- ---------------------------------------------------------------------------
-- 2.  Quintile scoring (1=worst, 5=best for each dimension)
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_rfm_scores;
CREATE VIEW v_rfm_scores AS
WITH scored AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,

        -- Recency: LOWER days → HIGHER score (inverted)
        CASE
            WHEN recency_days <= 30  THEN 5
            WHEN recency_days <= 60  THEN 4
            WHEN recency_days <= 120 THEN 3
            WHEN recency_days <= 180 THEN 2
            ELSE                          1
        END AS r_score,

        -- Frequency: more orders → higher score
        CASE
            WHEN frequency >= 20 THEN 5
            WHEN frequency >= 12 THEN 4
            WHEN frequency >= 6  THEN 3
            WHEN frequency >= 2  THEN 2
            ELSE                      1
        END AS f_score,

        -- Monetary: quintile-based
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score

    FROM v_rfm_raw
)
SELECT
    customer_id,
    recency_days,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    ROUND((r_score + f_score + m_score) / 3.0, 2) AS rfm_avg_score,
    CAST(r_score AS TEXT) || CAST(f_score AS TEXT) || CAST(m_score AS TEXT) AS rfm_cell
FROM scored;


-- ---------------------------------------------------------------------------
-- 3.  Rule-based RFM segment labels
--     (deterministic, interpretable — supplements ML clustering)
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_rfm_segments;
CREATE VIEW v_rfm_segments AS
SELECT
    *,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2                  THEN 'Promising'
        WHEN r_score >= 3 AND f_score >= 3 AND m_score <= 2 THEN 'Potential Loyalists'
        WHEN r_score <= 2 AND f_score >= 4                  THEN 'At Risk'
        WHEN r_score = 1  AND f_score >= 4                  THEN 'Cannot Lose Them'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score >= 3 THEN 'Hibernating High Value'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN 'Lost'
        ELSE                                                      'Need Attention'
    END AS rfm_segment
FROM v_rfm_scores;


-- ---------------------------------------------------------------------------
-- 4.  Segment summary for dashboarding
-- ---------------------------------------------------------------------------
/*
SELECT
    rfm_segment,
    COUNT(*)                        AS customer_count,
    ROUND(AVG(recency_days),  1)    AS avg_recency_days,
    ROUND(AVG(frequency),     1)    AS avg_orders,
    ROUND(AVG(monetary),      0)    AS avg_revenue,
    ROUND(SUM(monetary),      0)    AS total_revenue
FROM   v_rfm_segments
GROUP  BY rfm_segment
ORDER  BY total_revenue DESC;
*/
