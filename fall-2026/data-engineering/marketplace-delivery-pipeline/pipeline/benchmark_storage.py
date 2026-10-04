from __future__ import annotations

import shutil
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

import pandas as pd

from common import METRICS_DIR, STEP_DIRS, TS_FORMAT, ensure_dirs

SRC = STEP_DIRS["step2"] / "orders_clean.csv"
TS = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
      "order_delivered_customer_date", "order_estimated_delivery_date"]


def load_typed() -> pd.DataFrame:
    o = pd.read_csv(SRC, dtype={"customer_zip_code_prefix": "string"}, parse_dates=TS,
                    low_memory=False)
    o["delivered_after_estimate"] = o["delivered_after_estimate"].astype("Int8")
    return o


def types_kept(d: pd.DataFrame) -> dict:
    zip_ = d["customer_zip_code_prefix"].dropna()
    return {
        "zip_text_leading_zeros": bool(zip_.map(lambda v: isinstance(v, str) and len(v) == 5).all()),
        "timestamps_datetime": bool(all(pd.api.types.is_datetime64_any_dtype(d[c]) for c in TS)),
        "label_integer": bool(pd.api.types.is_integer_dtype(d["delivered_after_estimate"])),
        "flags_boolean": bool(pd.api.types.is_bool_dtype(d["delivery_fabricated"])),
    }


def main() -> None:
    ensure_dirs()
    df = load_typed()
    tmp = Path(tempfile.mkdtemp(prefix="eng2040_storage_"))
    fmts = {
        "csv": (lambda p: df.to_csv(p, index=False, date_format=TS_FORMAT), lambda p: pd.read_csv(p, low_memory=False), ".csv"),
        "csv_gzip": (lambda p: df.to_csv(p, index=False, date_format=TS_FORMAT, compression="gzip"),
                     lambda p: pd.read_csv(p, low_memory=False), ".csv.gz"),
        "sqlite_table": (lambda p: df.to_sql("orders_clean", sqlite3.connect(p), index=False),
                         lambda p: pd.read_sql("SELECT * FROM orders_clean", sqlite3.connect(p)), ".sqlite"),
        "parquet_snappy": (lambda p: df.to_parquet(p, index=False), lambda p: pd.read_parquet(p), ".parquet"),
        "feather_lz4": (lambda p: df.to_feather(p), lambda p: pd.read_feather(p), ".feather"),
    }
    rows, peak_tmp = [], 0
    for name, (write, read, ext) in fmts.items():
        path = tmp / f"orders_clean{ext}"
        # write once before timing so loading pyarrow is not counted
        write(tmp / f"warmup{ext}")
        (tmp / f"warmup{ext}").unlink()
        t0 = time.perf_counter()
        write(path)
        w = time.perf_counter() - t0
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            back = read(path)
            times.append(time.perf_counter() - t0)
        size = path.stat().st_size
        peak_tmp = max(peak_tmp, sum(p.stat().st_size for p in tmp.iterdir()))
        rows.append({"format": name, "status": "measured", "bytes": size,
                     "write_s": round(w, 4), "read_s_median_of_5": round(statistics.median(times), 4),
                     **types_kept(back)})
    shutil.rmtree(tmp)
    out = pd.DataFrame(rows)
    out["temporary_peak_bytes_all_formats"] = peak_tmp
    out.to_csv(METRICS_DIR / "storage_benchmark.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
