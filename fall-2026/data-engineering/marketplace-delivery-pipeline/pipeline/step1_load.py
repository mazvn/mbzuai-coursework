from __future__ import annotations

import sqlite3
import sys
import time

import pandas as pd

from common import DB_PATH, RAW_SQL, STEP_DIRS, df_bytes, file_bytes, step_metrics, write_json

OUT = STEP_DIRS["step1"]

# add up items and payments per order first, so the joins are one to one (no double counting)
ORDERS_SQL = """
WITH item_agg AS (
    SELECT order_id,
           COUNT(*)                      AS n_item_rows,
           ROUND(SUM(price), 2)          AS raw_price_sum,
           ROUND(SUM(freight_value), 2)  AS raw_freight_sum,
           SUM(price IS NULL)            AS n_price_null,
           SUM(freight_value IS NULL)    AS n_freight_null
    FROM order_items GROUP BY order_id
),
pay_agg AS (
    SELECT order_id,
           COUNT(*)                      AS n_payment_rows,
           SUM(payment_value)            AS raw_payment_centavos_sum,
           SUM(payment_value IS NULL)    AS n_payment_null,
           SUM(payment_value < 0)        AS n_payment_negative,
           SUM(ABS(payment_value))       AS raw_payment_abs_centavos_sum
    FROM order_payments GROUP BY order_id
)
SELECT o.order_id, o.customer_id, o.order_status,
       o.order_purchase_timestamp, o.order_approved_at,
       o.order_delivered_carrier_date, o.order_delivered_customer_date,
       o.order_estimated_delivery_date,
       c.customer_id IS NOT NULL         AS customer_found,
       c.customer_unique_id, c.customer_zip_code_prefix, c.customer_city, c.customer_state,
       COALESCE(i.n_item_rows, 0)        AS n_item_rows,
       i.raw_price_sum, i.raw_freight_sum,
       COALESCE(i.n_price_null, 0)       AS n_price_null,
       COALESCE(i.n_freight_null, 0)     AS n_freight_null,
       COALESCE(p.n_payment_rows, 0)     AS n_payment_rows,
       p.raw_payment_centavos_sum,
       COALESCE(p.n_payment_null, 0)     AS n_payment_null,
       COALESCE(p.n_payment_negative, 0) AS n_payment_negative,
       p.raw_payment_abs_centavos_sum
FROM orders o
LEFT JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN item_agg  i ON i.order_id    = o.order_id
LEFT JOIN pay_agg   p ON p.order_id    = o.order_id
ORDER BY o.order_id
"""

ITEMS_SQL = """
SELECT i.order_id, i.order_item_id, i.product_id, i.seller_id, i.price, i.freight_value,
       p.product_id IS NOT NULL AS product_found,
       p.product_category_name, t.product_category_name_english,
       p.product_name_lenght       AS product_name_length,
       p.product_description_lenght AS product_description_length,
       p.product_photos_qty, p.product_weight_g,
       p.product_length_cm, p.product_height_cm, p.product_width_cm,
       s.seller_id IS NOT NULL AS seller_found,
       s.seller_zip_code_prefix, s.seller_state
FROM order_items i
LEFT JOIN products p ON p.product_id = i.product_id
LEFT JOIN category_translation t
       ON t.product_category_name = lower(trim(p.product_category_name))
LEFT JOIN sellers s ON s.seller_id = i.seller_id
ORDER BY i.order_id, i.order_item_id
"""

ORDERS_DTYPES = {"customer_zip_code_prefix": "string", "n_item_rows": "int16",
                 "n_payment_rows": "int16", "customer_found": "bool"}
# hybrid version (joins in pandas), only used as an alternative in the loading benchmark
ITEM_QUERIES = {
    "items": "SELECT order_id, order_item_id, product_id, seller_id, price, freight_value FROM order_items",
    "products": "SELECT product_id, product_category_name, product_name_lenght AS product_name_length, "
                "product_description_lenght AS product_description_length, product_photos_qty, "
                "product_weight_g, product_length_cm, product_height_cm, product_width_cm FROM products",
    "translation": "SELECT product_category_name, product_category_name_english FROM category_translation",
    "sellers": "SELECT seller_id, seller_zip_code_prefix, seller_state FROM sellers",
}
ITEM_COLUMNS = ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value",
                "product_found", "product_category_name", "product_category_name_english",
                "product_name_length", "product_description_length", "product_photos_qty",
                "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
                "seller_found", "seller_zip_code_prefix", "seller_state"]


def load_items(con) -> pd.DataFrame:
    i = pd.read_sql(ITEM_QUERIES["items"], con, dtype={"order_item_id": "int16"})
    p = pd.read_sql(ITEM_QUERIES["products"], con)
    t = pd.read_sql(ITEM_QUERIES["translation"], con)
    s = pd.read_sql(ITEM_QUERIES["sellers"], con, dtype={"seller_zip_code_prefix": "string"})
    p["product_found"] = True
    p["key"] = p["product_category_name"].str.strip().str.lower()
    p = p.merge(t.rename(columns={"product_category_name": "key"}), on="key", how="left").drop(columns="key")
    s["seller_found"] = True
    items = i.merge(p, on="product_id", how="left").merge(s, on="seller_id", how="left")
    items["product_found"] = items["product_found"].fillna(False).astype(bool)
    items["seller_found"] = items["seller_found"].fillna(False).astype(bool)
    return items.sort_values(["order_id", "order_item_id"], kind="stable").reset_index(drop=True)[ITEM_COLUMNS]


ITEMS_DTYPES = {"seller_zip_code_prefix": "string", "order_item_id": "int16",
                "product_found": "bool", "seller_found": "bool"}

CHUNK_ROWS = 10_000


def read_sql_chunked(sql: str, con, dtype) -> pd.DataFrame:
    return pd.concat(pd.read_sql(sql, con, dtype=dtype, chunksize=CHUNK_ROWS), ignore_index=True)


# checks on keys and totals, saved to load_checks.json
CHECK_SQL = {
    "orders_rows": "SELECT COUNT(*) FROM orders",
    "orders_distinct_ids": "SELECT COUNT(DISTINCT order_id) FROM orders",
    "items_rows": "SELECT COUNT(*) FROM order_items",
    "payments_rows": "SELECT COUNT(*) FROM order_payments",
    "items_price_sum": "SELECT SUM(price) FROM order_items",
    "items_freight_sum": "SELECT SUM(freight_value) FROM order_items",
    "payments_centavos_sum": "SELECT SUM(payment_value) FROM order_payments",
    "items_orphan_order": "SELECT COUNT(*) FROM order_items WHERE order_id NOT IN (SELECT order_id FROM orders)",
    "payments_orphan_order": "SELECT COUNT(*) FROM order_payments WHERE order_id NOT IN (SELECT order_id FROM orders)",
    "reviews_orphan_order": "SELECT COUNT(*) FROM order_reviews WHERE order_id NOT IN (SELECT order_id FROM orders)",
    "orders_missing_customer": "SELECT COUNT(*) FROM orders o LEFT JOIN customers c USING(customer_id) WHERE c.customer_id IS NULL",
    "customers_without_order": "SELECT COUNT(*) FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders)",
    "customer_ids_shared_by_orders": "SELECT COUNT(*) FROM (SELECT customer_id FROM orders GROUP BY 1 HAVING COUNT(*) > 1)",
    "items_missing_product": "SELECT COUNT(*) FROM order_items i LEFT JOIN products p USING(product_id) WHERE p.product_id IS NULL",
    "items_missing_seller": "SELECT COUNT(*) FROM order_items i LEFT JOIN sellers s USING(seller_id) WHERE s.seller_id IS NULL",
    "orders_without_items": "SELECT COUNT(*) FROM orders WHERE order_id NOT IN (SELECT order_id FROM order_items)",
    "orders_without_payments": "SELECT COUNT(*) FROM orders WHERE order_id NOT IN (SELECT order_id FROM order_payments)",
    "item_sequence_gaps": "SELECT COUNT(*) FROM (SELECT order_id FROM order_items GROUP BY 1 HAVING MAX(order_item_id) <> COUNT(*))",
    "orphan_orders_matching_after_trim_lower": "SELECT COUNT(*) FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers) AND lower(trim(customer_id)) IN (SELECT lower(trim(customer_id)) FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders))",
    "orphan_orders_sharing_first_8_chars": "SELECT COUNT(*) FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers) AND substr(customer_id, 1, 8) IN (SELECT substr(customer_id, 1, 8) FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders))",
    "payment_type_raw_variants": "SELECT COUNT(DISTINCT payment_type) FROM order_payments",
    "payment_type_clean_variants": "SELECT COUNT(DISTINCT lower(trim(payment_type))) FROM order_payments",
    "payment_type_unclassified_rows": "SELECT COUNT(*) FROM order_payments WHERE lower(trim(payment_type)) = 'unclassified_payment'",
    "payment_installments_negative": "SELECT COUNT(*) FROM order_payments WHERE payment_installments < 0",
    "payment_installments_zero": "SELECT COUNT(*) FROM order_payments WHERE payment_installments = 0",
    "review_score_outside_1_5": "SELECT COUNT(*) FROM order_reviews WHERE review_score NOT BETWEEN 1 AND 5",
}


def string_storage() -> str:
    return getattr(pd.Series(["a"]).dtype, "storage", "python")


# guess the DataFrame size from SQL before loading: text cost depends on how pandas stores strings
def estimate_frame_bytes(con, sql: str) -> tuple[int, int]:
    cols = [d[0] for d in con.execute(f"SELECT * FROM ({sql}) LIMIT 0").description]
    exprs = ", ".join(
        f'SUM(typeof("{c}")=\'text\'), AVG(CASE WHEN typeof("{c}")=\'text\' THEN length("{c}") END), '
        f'SUM(typeof("{c}")=\'text\' OR (typeof("{c}")=\'null\'))'
        for c in cols)
    row = con.execute(f"SELECT COUNT(*), {exprs} FROM ({sql})").fetchone()
    n = row[0]
    arrow = string_storage() == "pyarrow"
    est = 0
    for k in range(len(cols)):
        n_text, avg_len = row[1 + 3 * k] or 0, row[2 + 3 * k] or 0
        if n_text == 0:
            est += n * 8
        elif arrow:
            est += n * 8 + n_text * avg_len + n // 8
        else:
            est += n * 8 + n_text * (41 + avg_len)
    return int(n), int(est + 128 * len(cols))


def main() -> int:
    with step_metrics("step1") as rec:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        checks = {k: con.execute(q).fetchone()[0] for k, q in CHECK_SQL.items()}

        n_o, est_o = estimate_frame_bytes(con, ORDERS_SQL)
        n_i, est_i = estimate_frame_bytes(con, ITEMS_SQL)

        t0 = time.perf_counter()
        orders = read_sql_chunked(ORDERS_SQL, con, ORDERS_DTYPES)
        items = read_sql_chunked(ITEMS_SQL, con, ITEMS_DTYPES)
        load_s = time.perf_counter() - t0

        # make sure nothing was lost or counted twice
        recon = {
            "orders_out_rows": len(orders), "orders_in_rows": checks["orders_rows"],
            "orders_out_unique_ids": int(orders["order_id"].nunique()),
            "items_out_rows": len(items), "items_in_rows": checks["items_rows"],
            "item_rows_via_order_aggregate": int(orders["n_item_rows"].sum()),
            "payment_rows_via_order_aggregate": int(orders["n_payment_rows"].sum()),
            "price_sum_in": checks["items_price_sum"],
            "price_sum_out_items": float(items["price"].sum()),
            "price_sum_out_orders": float(orders["raw_price_sum"].sum()),
            "freight_sum_in": checks["items_freight_sum"],
            "freight_sum_out_orders": float(orders["raw_freight_sum"].sum()),
            "payments_centavos_in": checks["payments_centavos_sum"],
            "payments_centavos_out": float(orders["raw_payment_centavos_sum"].sum()),
        }
        tol = 1e-6
        assert recon["orders_out_rows"] == recon["orders_in_rows"] == recon["orders_out_unique_ids"]
        assert recon["items_out_rows"] == recon["items_in_rows"] == recon["item_rows_via_order_aggregate"]
        assert recon["payment_rows_via_order_aggregate"] == checks["payments_rows"]
        for a, b in [("price_sum_in", "price_sum_out_items"), ("price_sum_in", "price_sum_out_orders"),
                     ("freight_sum_in", "freight_sum_out_orders"),
                     ("payments_centavos_in", "payments_centavos_out")]:
            assert abs(recon[a] - recon[b]) <= tol * max(1.0, abs(recon[a])), (a, b)
        recon["all_assertions_passed"] = True

        paths = [OUT / "orders_raw.csv", OUT / "items_raw.csv"]
        orders.to_csv(paths[0], index=False)
        items.to_csv(paths[1], index=False)
        write_json(OUT / "load_checks.json", {"relationship_checks": checks, "reconciliation": recon})
        (OUT / "executed_sql.sql").write_text(
            "-- orders_raw\n" + ORDERS_SQL + ";\n\n-- items_raw\n" + ITEMS_SQL + ";\n\n-- checks\n"
            + "\n".join(f"-- {k}\n{q};" for k, q in CHECK_SQL.items()) + "\n")

        rec["load_query_runtime_s"] = round(load_s, 3)
        con.close()

        rec.update({
            "inputs": {"sqlite_db_bytes": file_bytes([DB_PATH]),
                       "raw_sql_bytes": file_bytes([RAW_SQL])},
            "estimated_in_memory_bytes": {"orders_raw": est_o, "items_raw": est_i},
            "measured_in_memory_bytes": {"orders_raw": df_bytes(orders), "items_raw": df_bytes(items)},
            "rows": {"orders_raw": n_o, "items_raw": n_i},
            "pandas_string_storage": string_storage(),
            "output_bytes": {p.name: file_bytes([p]) for p in OUT.iterdir()},
            "temporary_storage_bytes": 0,
        })
    print(f"step1: {len(orders):,} orders, {len(items):,} item rows; checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
