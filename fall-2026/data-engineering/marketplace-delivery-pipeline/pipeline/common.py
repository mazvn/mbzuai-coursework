from __future__ import annotations

import json
import os
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("ENG2040_DB", ROOT / "marketplace.sqlite"))
OUT = Path(os.environ.get("ENG2040_OUT", ROOT / "outputs"))
RAW_SQL = ROOT / "data" / "raw.sql"
STEP_DIRS = {
    "step1": OUT / "step1_load",
    "step2": OUT / "step2_clean",
    "step3": OUT / "step3_eda",
    "step4": OUT / "step4_features",
}
METRICS_DIR = OUT / "metrics"

# placeholder values found while profiling
SENTINEL_TIMESTAMP = "2018-02-30 10:00:00"
SENTINEL_ZIP = "12A?0"
UNKNOWN_STATE = "ZZ"
UNMAPPED_CATEGORY = "unmapped_category"
FABRICATED_DELIVERY_OFFSET_S = 7 * 86400
# last day of the extract (orders due after this are only here if already delivered)
SNAPSHOT_DATE = "2018-08-31"
# each threshold sits in the gap between the highest real value and the smallest x1000 value
PRICE_SCALE_THRESHOLD = 6700.0
FREIGHT_SCALE_THRESHOLD = 500.0
WEIGHT_SCALE_THRESHOLD = 45000.0
DIM_SCALE_THRESHOLD = 1000.0
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
BRAZIL_STATES = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
}


def ensure_dirs() -> None:
    for d in [*STEP_DIRS.values(), METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def file_bytes(paths) -> int:
    return int(sum(Path(p).stat().st_size for p in paths if Path(p).exists()))


def df_bytes(df) -> int:
    return int(df.memory_usage(deep=True).sum())


@contextmanager
def step_metrics(step: str):
    ensure_dirs()
    record: dict = {"step": step}
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        record["runtime_s"] = round(time.perf_counter() - t0, 3)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record["python_heap_peak_bytes"] = int(peak)
        (METRICS_DIR / f"{step}.json").write_text(json.dumps(record, indent=2))


def _no_nan(x):
    if isinstance(x, dict):
        return {k: _no_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_no_nan(v) for v in x]
    if isinstance(x, float) and x != x:
        return None
    return x


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(_no_nan(obj), indent=2, default=str, allow_nan=False))
