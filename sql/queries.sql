-- Instamart Part A: seven analytical queries (SQLite >= 3.25). Only completed orders count as revenue/spend.
-- Each query is preceded by a "-- Qn" marker line; src/a4_sql.py splits the file on those markers.

-- Q1: WHERE (row filter) then HAVING (group filter)
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_over_500, COUNT(*) AS n_orders
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_over_500 DESC;

-- Q2: three-table join, revenue by city and product category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: scalar subquery, products priced above the average catalogue price
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN, customers with no completed order
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: CTEs, customers whose total completed spend is above the average customer total
WITH totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders WHERE status = 'completed' GROUP BY customer_id
), avg_total AS (
    SELECT AVG(total_spend) AS avg_spend FROM totals
)
SELECT t.customer_id, ROUND(t.total_spend, 2) AS total_spend, ROUND(a.avg_spend, 2) AS avg_spend
FROM totals t CROSS JOIN avg_total a
WHERE t.total_spend > a.avg_spend
ORDER BY t.total_spend DESC;

-- Q6: ranking window functions within each city
WITH spend AS (
    SELECT c.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, row_num;

-- Q7: daily completed revenue with a running (cumulative) total
WITH daily AS (
    SELECT substr(order_ts, 1, 10) AS day, ROUND(SUM(order_amount), 2) AS revenue
    FROM orders WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day, revenue, ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
