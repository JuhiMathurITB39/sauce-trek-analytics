-- =============================================================================
-- 01_data_cleaning.sql
-- Sauce Trek Analytics — Data Cleaning Layer
-- Compatible with: SQLite / PostgreSQL / MySQL (minor dialect tweaks noted)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1.  Remove exact duplicate orders
-- ---------------------------------------------------------------------------
DELETE FROM orders
WHERE rowid NOT IN (                        -- SQLite; use ctid in Postgres
    SELECT MIN(rowid)
    FROM   orders
    GROUP  BY order_id, customer_id, order_date, total_amount
);

-- ---------------------------------------------------------------------------
-- 2.  Validate & cap discount_percentage (must be 0–100)
-- ---------------------------------------------------------------------------
UPDATE orders
SET    discount_percentage = 0
WHERE  discount_percentage < 0
   OR  discount_percentage > 100;

-- Keep discount_applied flag consistent with percentage
UPDATE orders
SET    discount_applied = 'No',
       discount_percentage = 0
WHERE  discount_applied = 'Yes'
  AND  discount_percentage = 0;

UPDATE orders
SET    discount_applied = 'Yes'
WHERE  discount_applied = 'No'
  AND  discount_percentage > 0;

-- ---------------------------------------------------------------------------
-- 3.  Remove orders with non-positive total
-- ---------------------------------------------------------------------------
DELETE FROM orders
WHERE  total_amount <= 0;

-- ---------------------------------------------------------------------------
-- 4.  Remove orphaned order_items (no matching order)
-- ---------------------------------------------------------------------------
DELETE FROM order_items
WHERE  order_id NOT IN (SELECT order_id FROM orders);

-- ---------------------------------------------------------------------------
-- 5.  Remove order_items with bad quantity / price
-- ---------------------------------------------------------------------------
DELETE FROM order_items
WHERE  quantity <= 0
   OR  price    <= 0;

-- ---------------------------------------------------------------------------
-- 6.  Remove customers with no orders (cold-start / test accounts)
--     (Run AFTER cleaning orders so real zero-order customers are captured)
-- ---------------------------------------------------------------------------
-- CREATE TABLE customers_active AS          -- uncomment to materialise
SELECT c.*
FROM   customers c
WHERE  EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
);

-- ---------------------------------------------------------------------------
-- 7.  Standardise city names  (trim whitespace, title-case)
--     SQLite uses TRIM; Postgres has INITCAP()
-- ---------------------------------------------------------------------------
UPDATE customers
SET    city = TRIM(city);

-- ---------------------------------------------------------------------------
-- 8.  Flag orders outside plausible date range
--     (before company founding 2022-01-01 or after data extract 2024-06-30)
-- ---------------------------------------------------------------------------
ALTER TABLE orders ADD COLUMN date_flag INTEGER DEFAULT 0;

UPDATE orders
SET    date_flag = 1
WHERE  order_date < '2022-01-01'
   OR  order_date > '2024-06-30';

-- Review flagged rows:
-- SELECT * FROM orders WHERE date_flag = 1;

-- ---------------------------------------------------------------------------
-- 9.  Confirm referential integrity: every order belongs to a known customer
-- ---------------------------------------------------------------------------
-- SELECT COUNT(*) AS orphaned_orders
-- FROM   orders o
-- LEFT   JOIN customers c ON o.customer_id = c.customer_id
-- WHERE  c.customer_id IS NULL;

-- ---------------------------------------------------------------------------
-- 10. Create cleaned views for downstream analytics
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_orders_clean;
CREATE VIEW v_orders_clean AS
SELECT *
FROM   orders
WHERE  date_flag = 0
  AND  total_amount > 0;

DROP VIEW IF EXISTS v_items_clean;
CREATE VIEW v_items_clean AS
SELECT oi.*
FROM   order_items oi
JOIN   v_orders_clean o ON oi.order_id = o.order_id
WHERE  oi.quantity > 0
  AND  oi.price    > 0;
