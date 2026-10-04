from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
WORK_DB = OUT / "step0_database" / "marketplace.sqlite"
RAW_SQL = ROOT / "data" / "raw.sql"
STEPS = [("step1", "pipeline/step1_load.py", "step1_load"),
         ("step2", "pipeline/step2_clean.py", "step2_clean"),
         ("step3", "pipeline/step3_eda.py", "step3_eda"),
         ("step4", "pipeline/step4_features.py", "step4_features")]
PILOT_FRACTIONS = [0.10, 0.20, 0.40]
BENCH_REPEATS = 5


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(d: Path) -> int:
    total = 0
    for p in Path(d).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            pass
    return total


class DiskPoller:
    # checks the folder size every 50 ms; peak minus final size = temporary files
    def __init__(self, folder: Path):
        self.folder, self.peak, self._stop = folder, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, dir_bytes(self.folder))
            time.sleep(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, dir_bytes(self.folder))


def run(cmd, env, quiet: bool = False) -> tuple[int, float, int | None]:
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL if quiet else None)
    # wait4 gives the peak memory (RSS) of the finished child process
    _, status, ru = os.wait4(proc.pid, 0)
    code = os.waitstatus_to_exitcode(status)
    rss_kib = ru.ru_maxrss if sys.platform != "darwin" else ru.ru_maxrss // 1024
    return code, time.perf_counter() - t0, rss_kib * 1024


def run_steps(db: Path, out: Path, quiet: bool) -> list[dict]:
    env = {**os.environ, "ENG2040_DB": str(db), "ENG2040_OUT": str(out), "PYTHONHASHSEED": "0"}
    rows = []
    for step, script, folder in STEPS:
        if not quiet:
            print(f"--- {step}: {script}", flush=True)
        with DiskPoller(out) as dp:
            before = dir_bytes(out)
            code, secs, rss = run([sys.executable, script], env, quiet)
        if code:
            raise SystemExit(f"{step} failed (exit {code}) for database {db}")
        m = json.loads((out / "metrics" / f"{step}.json").read_text())
        rows.append({"step": step, "runtime_s": round(secs, 3), "peak_rss_bytes": rss,
                     "python_heap_peak_bytes": m["python_heap_peak_bytes"],
                     "saved_output_bytes": dir_bytes(out / folder),
                     "temporary_bytes": max(0, dp.peak - dir_bytes(out)) if dp.peak > before else 0})
    return rows


def make_sample_db(src: Path, dst: Path, fraction: float) -> int:
    if dst.exists():
        dst.unlink()
    con = sqlite3.connect(dst)
    con.execute(f"ATTACH DATABASE '{src}' AS s")
    for (sql,) in con.execute("SELECT sql FROM s.sqlite_master WHERE type='table' AND sql IS NOT NULL"):
        con.execute(sql)
    n_all = con.execute("SELECT COUNT(*) FROM s.orders").fetchone()[0]
    con.execute(f"CREATE TEMP TABLE pick AS SELECT order_id, customer_id FROM s.orders "
                f"ORDER BY substr(order_id, 7) LIMIT {int(n_all * fraction)}")
    con.execute("INSERT INTO orders SELECT * FROM s.orders WHERE order_id IN (SELECT order_id FROM pick)")
    con.execute("INSERT INTO customers SELECT * FROM s.customers WHERE customer_id IN (SELECT customer_id FROM pick)")
    for t in ["order_items", "order_payments", "order_reviews"]:
        con.execute(f"INSERT INTO {t} SELECT * FROM s.{t} WHERE order_id IN (SELECT order_id FROM pick)")
    for t in ["products", "sellers", "category_translation"]:
        con.execute(f"INSERT INTO {t} SELECT * FROM s.{t}")
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    con.close()
    return n


# pilot runs on 10/20/40% of orders, then a straight line to 100%
def estimate(full_orders: int) -> tuple[list[dict], int]:
    tmp = Path(tempfile.mkdtemp(prefix="eng2040_pilot_"))
    pilots = []
    try:
        with DiskPoller(tmp) as dp:
            for f in PILOT_FRACTIONS:
                db = tmp / f"sample_{int(f * 100)}.sqlite"
                n = make_sample_db(WORK_DB, db, f)
                print(f"--- pilot run on {n:,} orders ({f:.0%})", flush=True)
                pilots.append((n, run_steps(db, tmp / f"out_{int(f * 100)}", quiet=True)))
        pilot_temp_peak = dp.peak
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pilot_rows = []
    for n, rows in pilots:
        for r in rows:
            pilot_rows.append({"orders": n, **r})
    write_csv(OUT / "metrics" / "pilot_runs.csv", [{k: ("" if v is None else v) for k, v in r.items()} for r in pilot_rows])
    est = []
    for step in [r["step"] for r in pilots[0][1]]:
        row = {"step": step}
        for k in ["runtime_s", "peak_rss_bytes", "python_heap_peak_bytes", "saved_output_bytes"]:
            pts = [(n, next(r[k] for r in rows if r["step"] == step)) for n, rows in pilots]
            if any(v is None for _, v in pts):
                row[k] = None
                continue
            xs = [p[0] for p in pts]
            ys = [float(p[1]) for p in pts]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            pred = my + slope * (full_orders - mx)
            # never predict less than the biggest pilot run
            row[k] = max(pred, max(ys))
        temps = [next(r["temporary_bytes"] for r in rows if r["step"] == step) for _, rows in pilots]
        row["temporary_bytes"] = 0 if all(t == 0 for t in temps) else max(temps) * full_orders / pilots[-1][0]
        est.append(row)
    return est, pilot_temp_peak


def make_scaled_db(src: Path, dst: Path, times: int) -> None:
    # copy every order (with its customer, items, payments and reviews) `times` times
    # with a suffix on the ids, so the loading test can be repeated on bigger data
    if dst.exists():
        dst.unlink()
    con = sqlite3.connect(dst)
    con.execute(f"ATTACH DATABASE '{src}' AS s")
    for (sql,) in con.execute("SELECT sql FROM s.sqlite_master WHERE type='table' AND sql IS NOT NULL"):
        con.execute(sql)
    for k in range(times):
        sfx = f"'-{k}'"
        con.execute(f"INSERT INTO orders SELECT order_id || {sfx}, customer_id || {sfx}, order_status, "
                    "order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, "
                    "order_delivered_customer_date, order_estimated_delivery_date FROM s.orders")
        con.execute(f"INSERT INTO customers SELECT customer_id || {sfx}, customer_unique_id, "
                    "customer_zip_code_prefix, customer_city, customer_state FROM s.customers")
        con.execute(f"INSERT INTO order_items SELECT order_id || {sfx}, order_item_id, product_id, seller_id, "
                    "shipping_limit_date, price, freight_value FROM s.order_items")
        con.execute(f"INSERT INTO order_payments SELECT order_id || {sfx}, payment_sequential, payment_type, "
                    "payment_installments, payment_value FROM s.order_payments")
        con.execute(f"INSERT INTO order_reviews SELECT review_id || {sfx}, order_id || {sfx}, review_score, "
                    "review_comment_title, review_comment_message, review_creation_date, "
                    "review_answer_timestamp FROM s.order_reviews")
    for t in ["products", "sellers", "category_translation"]:
        con.execute(f"INSERT INTO {t} SELECT * FROM s.{t}")
    con.commit()
    con.close()


def bench_round(env, variants, repeats, tag):
    bench = OUT / "metrics" / "bench"
    rows = []
    for rep in range(1, repeats + 1):
        for v in variants:
            name = f"{tag}{rep}"
            code, secs, rss = run([sys.executable, "pipeline/benchmark_loading.py", v, name], env)
            r = json.loads((bench / f"loading_{v}_{name}.json").read_text())
            r["process_peak_rss_bytes"] = rss
            rows.append(r)
    summary = []
    for v in variants:
        rr = [r for r in rows if r["variant"] == v]
        summary.append({"data_size": "1x" if not tag else tag.rstrip("_"), "variant": v, "runs": len(rr),
                        "result_rows": rr[0]["result_rows"],
                        "one_row_per_order": rr[0]["rows_one_per_order"],
                        "price_total": round(rr[0]["price_total"], 2),
                        "rows_read_into_python": rr[0]["rows_read_into_python"],
                        "runtime_s_median": round(statistics.median(r["runtime_s"] for r in rr), 4),
                        "runtime_s_min": min(r["runtime_s"] for r in rr),
                        "runtime_s_max": max(r["runtime_s"] for r in rr),
                        "python_heap_peak_bytes_median": int(statistics.median(r["python_heap_peak_bytes"] for r in rr)),
                        "process_peak_rss_bytes_median": int(statistics.median(r["process_peak_rss_bytes"] for r in rr)),
                        "final_in_memory_bytes": rr[0]["final_in_memory_bytes"]})
    return summary


def benchmarks(env) -> None:
    shutil.rmtree(OUT / "metrics" / "bench", ignore_errors=True)
    summary = bench_round(env, ["naive", "pandas_agg", "hybrid", "chosen"], BENCH_REPEATS, "")
    # same test on 5x the data, only for the three correct methods
    tmp = Path(tempfile.mkdtemp(prefix="eng2040_scaled_"))
    try:
        big = tmp / "scaled_5x.sqlite"
        make_scaled_db(WORK_DB, big, 5)
        summary += bench_round({**env, "ENG2040_DB": str(big)}, ["pandas_agg", "hybrid", "chosen"], 3, "5x_")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    write_csv(OUT / "metrics" / "loading_benchmark.csv", summary)
    run([sys.executable, "pipeline/benchmark_storage.py"], env, quiet=True)


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "machine": platform.machine(), "cpu_count": os.cpu_count(),
           "sqlite": sqlite3.sqlite_version}
    try:
        import pandas as pd
        env["pandas_string_storage"] = getattr(pd.Series(["a"]).dtype, "storage", "python")
    except ImportError:
        pass
    for pkg in ["pandas", "numpy", "matplotlib", "sklearn", "pyarrow"]:
        try:
            env[pkg] = __import__(pkg).__version__
        except ImportError:
            env[pkg] = "not installed"
    return env


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = set(sys.argv[1:])
    raw_hash_before = sha256(RAW_SQL)
    for sub in ["step0_database", "step1_load", "step2_clean", "step3_eda", "step4_features", "metrics"]:
        shutil.rmtree(OUT / sub, ignore_errors=True)
    (OUT / "step0_database").mkdir(parents=True)
    (OUT / "metrics").mkdir(parents=True)
    env = {**os.environ, "ENG2040_DB": str(WORK_DB), "ENG2040_OUT": str(OUT), "PYTHONHASHSEED": "0"}

    print("--- step0: build working database", flush=True)
    code, secs, rss = run([sys.executable, "setup_db.py", "--db", str(WORK_DB)], env)
    if code:
        return code
    n_orders = sqlite3.connect(WORK_DB).execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    measured = [{"step": "step0", "runtime_s": round(secs, 3), "peak_rss_bytes": rss,
                 "python_heap_peak_bytes": None, "saved_output_bytes": WORK_DB.stat().st_size,
                 "temporary_bytes": 0}]

    est, pilot_tmp = ([], 0) if "--skip-estimates" in args else estimate(n_orders)
    measured += run_steps(WORK_DB, OUT, quiet=False)

    code, secs, base_rss = run([sys.executable, "-c", "import pandas, numpy, matplotlib, sqlite3"], env)
    if "--skip-benchmarks" not in args:
        print("--- benchmarks (loading x5, storage formats)", flush=True)
        benchmarks(env)

    notes = {"step0": "working SQLite copy: intermediate, may be deleted after step 1",
             "step1": "", "step2": "", "step3": "", "step4": "includes scikit-learn leakage check"}
    res = [{**m, "note": notes[m["step"]]} for m in measured]
    res.append({"step": "baseline_imports", "runtime_s": round(secs, 3), "peak_rss_bytes": base_rss,
                "python_heap_peak_bytes": None, "saved_output_bytes": 0, "temporary_bytes": 0,
                "note": "Python + pandas/numpy/matplotlib imported, no data (fixed RSS overhead)"})
    if est:
        res.append({"step": "pilot_runs", "runtime_s": None, "peak_rss_bytes": None,
                    "python_heap_peak_bytes": None, "saved_output_bytes": 0,
                    "temporary_bytes": pilot_tmp, "note": "temporary folder for estimate pilots, deleted"})
    write_csv(OUT / "metrics" / "resources.csv", [{k: ("" if v is None else v) for k, v in r.items()} for r in res])
    if est:
        cmp_rows = []
        for e in est:
            m = next(x for x in measured if x["step"] == e["step"])
            for k in ["runtime_s", "peak_rss_bytes", "python_heap_peak_bytes", "saved_output_bytes",
                      "temporary_bytes"]:
                ev, mv = e[k], m[k]
                cmp_rows.append({"step": e["step"], "metric": k,
                                 "estimated": "" if ev is None else round(ev, 3),
                                 "measured": "" if mv is None else mv,
                                 "measured_over_estimated": round(mv / ev, 3) if ev and mv else ""})
        write_csv(OUT / "metrics" / "resources_estimated_vs_measured.csv", cmp_rows)

    assert sha256(RAW_SQL) == raw_hash_before, "raw.sql changed!"
    envd = environment()
    envd["raw_sql_sha256"] = raw_hash_before
    (OUT / "metrics" / "environment.json").write_text(json.dumps(envd, indent=2))
    with open(OUT / "metrics" / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "bytes", "sha256"])
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and "metrics" not in p.parts and p.suffix != ".sqlite":
                w.writerow([p.relative_to(ROOT).as_posix(), p.stat().st_size, sha256(p)])
    for r in res:
        rss = r["peak_rss_bytes"]
        rss_txt = f"{rss / 2**20:7.1f} MiB" if rss else "      n/a"
        rt = f"{r['runtime_s']:>7}s" if r["runtime_s"] is not None else "        "
        print(f"{r['step']:>16}  {rt}  peak RSS {rss_txt}")
    print("raw.sql unchanged (sha256 verified). Outputs in", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
