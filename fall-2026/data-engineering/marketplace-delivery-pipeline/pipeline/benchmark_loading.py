from __future__ import annotations

import sqlite3
import sys
import time
import tracemalloc

import pandas as pd

from common import DB_PATH, METRICS_DIR, df_bytes, ensure_dirs, write_json
from step1_load import ITEMS_DTYPES, ITEMS_SQL, ITEM_QUERIES, ORDERS_DTYPES, ORDERS_SQL, load_items, read_sql_chunked

TABLES = ["orders", "customers", "order_items", "order_payments", "order_reviews",
          "products", "sellers", "category_translation"]


# the wrong way on purpose: joining items and payments directly multiplies rows
def naive(con):
    t = {n: pd.read_sql(f"SELECT * FROM {n}", con) for n in TABLES}
    fan = (t["orders"].merge(t["customers"], on="customer_id", how="left")
           .merge(t["order_items"], on="order_id", how="left")
           .merge(t["order_payments"], on="order_id", how="left"))
    held = sum(df_bytes(d) for d in t.values()) + df_bytes(fan)
    return {"rows_one_per_order": False, "rows_read_into_python": int(sum(len(d) for d in t.values())), "result_rows": len(fan), "final_in_memory_bytes": held,
            "price_total": float(fan["price"].sum())}


def pandas_agg(con):
    o = pd.read_sql("SELECT * FROM orders", con)
    c = pd.read_sql("SELECT customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, "
                    "customer_state FROM customers", con, dtype={"customer_zip_code_prefix": "string"})
    i = pd.read_sql("SELECT order_id, order_item_id, product_id, seller_id, price, freight_value "
                    "FROM order_items", con)
    p = pd.read_sql("SELECT order_id, payment_value FROM order_payments", con)
    pr = pd.read_sql("SELECT * FROM products", con)
    s = pd.read_sql("SELECT seller_id, seller_zip_code_prefix, seller_state FROM sellers", con,
                    dtype={"seller_zip_code_prefix": "string"})
    tr = pd.read_sql("SELECT * FROM category_translation", con)
    ia = i.groupby("order_id").agg(n_item_rows=("order_item_id", "size"),
                                   raw_price_sum=("price", "sum"), raw_freight_sum=("freight_value", "sum"))
    pa = p.groupby("order_id").agg(n_payment_rows=("payment_value", "size"),
                                   raw_payment_centavos_sum=("payment_value", "sum"))
    orders = (o.merge(c, on="customer_id", how="left", validate="1:1")
              .merge(ia, left_on="order_id", right_index=True, how="left", validate="1:1")
              .merge(pa, left_on="order_id", right_index=True, how="left", validate="1:1"))
    pr["k"] = pr["product_category_name"].str.strip().str.lower()
    items = (i.merge(pr.merge(tr, left_on="k", right_on="product_category_name", how="left",
                              suffixes=("", "_t")), on="product_id", how="left")
             .merge(s, on="seller_id", how="left"))
    return {"rows_one_per_order": True, "result_rows": len(orders),
            "rows_read_into_python": int(sum(len(d) for d in [o, c, i, p, pr, s, tr])),
            "final_in_memory_bytes": df_bytes(orders) + df_bytes(items),
            "price_total": float(orders["raw_price_sum"].sum())}


# the version used in step 1: totals and joins both done in SQLite, read in chunks
def chosen(con):
    o = read_sql_chunked(ORDERS_SQL, con, ORDERS_DTYPES)
    i = read_sql_chunked(ITEMS_SQL, con, ITEMS_DTYPES)
    return {"rows_one_per_order": True, "result_rows": len(o), "rows_read_into_python": len(o) + len(i),
            "final_in_memory_bytes": df_bytes(o) + df_bytes(i),
            "price_total": float(o["raw_price_sum"].sum())}


# alternative: totals in SQLite, product and seller details joined in pandas
def hybrid(con):
    o = read_sql_chunked(ORDERS_SQL, con, ORDERS_DTYPES)
    i = load_items(con)
    lookups = sum(con.execute(f"SELECT COUNT(*) FROM ({q})").fetchone()[0]
                  for k, q in ITEM_QUERIES.items() if k != "items")
    return {"rows_one_per_order": True, "result_rows": len(o),
            "rows_read_into_python": len(o) + len(i) + lookups,
            "final_in_memory_bytes": df_bytes(o) + df_bytes(i),
            "price_total": float(o["raw_price_sum"].sum())}


if __name__ == "__main__":
    variant, rep = sys.argv[1], sys.argv[2]
    ensure_dirs()
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    tracemalloc.start()
    t0 = time.perf_counter()
    res = {"naive": naive, "pandas_agg": pandas_agg, "hybrid": hybrid, "chosen": chosen}[variant](con)
    res["runtime_s"] = round(time.perf_counter() - t0, 4)
    res["python_heap_peak_bytes"] = int(tracemalloc.get_traced_memory()[1])
    res.update({"variant": variant, "repeat": rep})
    (METRICS_DIR / "bench").mkdir(exist_ok=True)
    write_json(METRICS_DIR / "bench" / f"loading_{variant}_{rep}.json", res)
