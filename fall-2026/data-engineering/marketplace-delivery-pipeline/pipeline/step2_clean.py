from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common import (METRICS_DIR, BRAZIL_STATES, DIM_SCALE_THRESHOLD, FABRICATED_DELIVERY_OFFSET_S,
                    FREIGHT_SCALE_THRESHOLD, PRICE_SCALE_THRESHOLD, SENTINEL_TIMESTAMP,
                    SENTINEL_ZIP, STEP_DIRS, TS_FORMAT, UNMAPPED_CATEGORY,
                    WEIGHT_SCALE_THRESHOLD, df_bytes, file_bytes, step_metrics, write_json)

IN, OUT = STEP_DIRS["step1"], STEP_DIRS["step2"]
TS_COLS = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
           "order_delivered_customer_date", "order_estimated_delivery_date"]
LOG: list[dict] = []


def log(rule, table, column, issue, evidence, action, n, ids=None):
    LOG.append({"rule": rule, "table": table, "column": column, "issue": issue,
                "evidence": evidence, "action": action, "rows_affected": int(n),
                "distinct_entities_affected": int(ids.nunique()) if ids is not None else int(n)})


def profile(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        is_text = s.dtype == object or str(s.dtype) in ("string", "str")
        row = {"table": name, "column": c, "dtype": str(s.dtype), "rows": len(s),
               "nulls": int(s.isna().sum()), "distinct": int(s.nunique())}
        if is_text:
            txt = s.dropna().astype(str)
            row.update({"min": txt.min() if len(txt) else None, "max": txt.max() if len(txt) else None,
                        "untrimmed_text": int((txt != txt.str.strip()).sum()),
                        "sentinel_timestamp": int((txt == SENTINEL_TIMESTAMP).sum()),
                        "sentinel_zip": int((txt == SENTINEL_ZIP).sum())})
        elif s.dtype.kind in "iufb" or pd.api.types.is_datetime64_any_dtype(s):
            row.update({"min": s.min(), "max": s.max(),
                        "negative": int((s < 0).sum()) if s.dtype.kind in "iuf" else None})
        rows.append(row)
    return pd.DataFrame(rows)


def clean_state(state, zip_, zip3_map, table, rule, ids, zip_evidence):
    raw = state.copy()
    st = state.str.strip().str.upper()
    n_fmt = int(((raw != st) & raw.notna()).sum())
    log(rule + "a", table, "state", "untrimmed / lower-case state codes",
        "e.g. ' sp  ' vs 'SP'", "strip + upper-case", n_fmt, ids[(raw != st) & raw.notna()])
    invalid = st.notna() & ~st.isin(BRAZIL_STATES)
    log(rule + "b", table, "state", "placeholder state 'ZZ' (not a Brazilian state)",
        "not one of the 27 UF codes", "set missing, then try zip recovery", invalid.sum(), ids[invalid])
    st = st.where(~invalid)
    z3 = zip_.str[:3]
    inferred = z3.map(zip3_map)
    need = st.isna()
    can = need & inferred.notna()
    log(rule + "c", table, "state", "state missing or placeholder",
        zip_evidence,
        "recover state from zip prefix", can.sum(), ids[can])
    source = np.where(~need, "original", np.where(can, "zip_inferred", "unknown"))
    st = st.fillna(inferred)
    return st, pd.Series(source, index=state.index)


def zip_map_from(z):
    counts = z.groupby(["z3", "state"]).size().rename("n").reset_index()
    per = counts.groupby("z3")["state"].nunique()
    unambiguous = per[per == 1].index
    return counts[counts.z3.isin(unambiguous)].set_index("z3")["state"].to_dict(), counts, per, unambiguous


# zip prefix -> state, only from prefixes that point to one state; tested on a held-out half
def build_zip3_map(frames):
    z = pd.concat(frames, ignore_index=True).dropna()
    z = z[z["state"].isin(BRAZIL_STATES) & z["zip"].str.fullmatch(r"\d{5}")].reset_index(drop=True)
    z["z3"] = z["zip"].str[:3]
    m, counts, per, unambiguous = zip_map_from(z)
    half = np.random.default_rng(2040).random(len(z)) < 0.5
    train_map = zip_map_from(z[half])[0]
    test = z[~half]
    pred = test["z3"].map(train_map)
    covered = pred.notna()
    stats = {"zip3_prefixes": int(per.size), "unambiguous_prefixes": int(len(unambiguous)),
             "entities_used": int(len(z)),
             "row_weighted_purity": float(counts.groupby("z3")["n"].max().sum() / len(z)),
             "holdout_test_entities": int(len(test)),
             "holdout_covered_share": float(covered.mean()),
             "holdout_accuracy": float((pred[covered] == test.loc[covered, "state"]).mean())}
    return m, stats


# numbers that back up the x1000 and minus rules (saved in validation.json)
def amount_evidence(it, col, thr):
    v = it[col]
    normal = v[(v > 0) & (v <= thr)]
    med = normal.groupby(it.loc[normal.index, "product_id"]).median()
    m = it["product_id"].map(med)
    neg = (v < 0) & m.notna()
    big = (v > thr) & m.notna()
    return {
        "highest_value_below_threshold": float(v.abs()[v.abs() <= thr].max()),
        "lowest_value_above_threshold": float(v.abs()[v.abs() > thr].min()),
        "negatives_with_known_product_value": int(neg.sum()),
        "negatives_exactly_minus_product_median": float((v[neg].abs().round(2) == m[neg].round(2)).mean()),
        "inflated_with_known_product_value": int(big.sum()),
        "inflated_exactly_1000x_product_median": float(((v[big] / m[big]).round(2) == 1000).mean()),
    }


def clean_items(items: pd.DataFrame, zip3_map, zip_evidence, evidence) -> pd.DataFrame:
    it = items.copy()
    for col, thr, rid in [("price", PRICE_SCALE_THRESHOLD, "I1"),
                          ("freight_value", FREIGHT_SCALE_THRESHOLD, "I2")]:
        ev = amount_evidence(it, col, thr)
        evidence[col] = ev
        neg = it[col] < 0
        log(rid + "a", "items", col, "negative amounts",
            f"{ev['negatives_exactly_minus_product_median']:.0%} of the {ev['negatives_with_known_product_value']} "
            "with a known usual price are exactly minus it; fixed orders match their payments",
            "absolute value", neg.sum())
        it[col] = it[col].abs()
        big = it[col] > thr
        log(rid + "b", "items", col, f"values inflated x1000 (> {thr:,.0f})",
            f"no values between {ev['highest_value_below_threshold']:,.2f} and {ev['lowest_value_above_threshold']:,.2f}; "
            f"{ev['inflated_exactly_1000x_product_median']:.0%} of the {ev['inflated_with_known_product_value']} with a "
            "known usual price are exactly 1000x it; fixed orders match their payments", "divide by 1000", big.sum())
        it[col] = it[col].where(~big, it[col] / 1000)
        it[f"{col}_repaired"] = neg | big

    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        thr = WEIGHT_SCALE_THRESHOLD if col == "product_weight_g" else DIM_SCALE_THRESHOLD
        neg = it[col] < 0
        it[col] = it[col].abs()
        big = it[col] > thr
        mult = float((it.loc[big, col] % 1000 == 0).mean()) if big.any() else 1.0
        below = float(it.loc[it[col] <= thr, col].max())
        above = float(it.loc[big, col].min()) if big.any() else float("nan")
        it[col] = it[col].where(~big, it[col] / 1000)
        log("P1", "items(product)", col, "negative or x1000 dimensions",
            f"no values between {below:,.0f} and {above:,.0f}; {mult:.0%} of values above {thr:,.0f} are multiples of 1000",
            "absolute value; divide x1000 values by 1000", (neg | big).sum(), it.loc[neg | big, "product_id"])
    for col in ["product_name_length", "product_description_length", "product_photos_qty"]:
        m1 = it[col] == -1
        negs = sorted(it.loc[it[col] < 0, col].unique().tolist())
        log("P2", "items(product)", col, "-1 placeholder", f"negative values present: {negs}",
            "set missing (optional field)", m1.sum(), it.loc[m1, "product_id"])
        it[col] = it[col].where(~m1)

    cat = it["product_category_name"].str.strip().str.lower()
    log("P3", "items(product)", "product_category_name", "untrimmed / upper-case categories",
        "e.g. '  AGRO_INDUSTRIA_E_COMERCIO '", "strip + lower-case",
        ((cat != it["product_category_name"]) & cat.notna()).sum(),
        it.loc[(cat != it["product_category_name"]) & cat.notna(), "product_id"])
    um = cat == UNMAPPED_CATEGORY
    log("P4", "items(product)", "product_category_name", "'unmapped_category' placeholder",
        "not a real category, not in the translation table", "set missing", um.sum(), it.loc[um, "product_id"])
    cat = cat.where(~um)
    it["product_category_name"] = cat
    no_en = cat.notna() & it["product_category_name_english"].isna()
    log("P5", "items(product)", "product_category_name_english",
        "category without English translation",
        "translation table has no English name for: " + ", ".join(sorted(cat[no_en].dropna().unique())),
        "use Portuguese name", no_en.sum(), it.loc[no_en, "product_id"])
    it["category"] = it["product_category_name_english"].fillna(cat)
    log("P6", "items", "product_id", "product not in products table",
        "id not found (step 1 check)", "keep item; product attributes missing",
        (~it["product_found"]).sum(), it.loc[~it["product_found"], "product_id"])
    log("S1", "items", "seller_id", "seller not in sellers table",
        "id not found (step 1 check)", "keep item; seller attributes missing",
        (~it["seller_found"]).sum(), it.loc[~it["seller_found"], "seller_id"])
    sz = it["seller_zip_code_prefix"]
    bad_zip = sz == SENTINEL_ZIP
    log("S2", "items(seller)", "seller_zip_code_prefix", "placeholder zip '12A?0'",
        "non-numeric; real prefixes are 5 digits", "set missing", bad_zip.sum(), it.loc[bad_zip, "seller_id"])
    it["seller_zip_code_prefix"] = sz.where(~bad_zip)
    it["seller_state"], it["seller_state_source"] = clean_state(
        it["seller_state"], it["seller_zip_code_prefix"], zip3_map, "items(seller)", "S3", it["seller_id"], zip_evidence)
    it.loc[~it["seller_found"], "seller_state_source"] = "seller_not_found"
    it["volume_cm3"] = it["product_length_cm"] * it["product_height_cm"] * it["product_width_cm"]
    return it


# missing price = payment - other amounts; tested first on orders where the answer is known
def recover_amounts(it: pd.DataFrame, orders: pd.DataFrame):
    pay = orders.set_index("order_id")
    pay_brl = pay["raw_payment_abs_centavos_sum"] / 100
    pay_complete = pay["n_payment_null"] == 0
    miss = it["price"].isna().astype(int) + it["freight_value"].isna().astype(int)
    n_miss = miss.groupby(it["order_id"]).transform("sum")
    known = (it["price"].fillna(0) + it["freight_value"].fillna(0)).groupby(it["order_id"]).transform("sum")

    comp_ids = pay.index[pay_complete]
    comp = it[(n_miss == 0) & it["order_id"].isin(comp_ids)]
    rng = np.random.default_rng(2040)
    pick = (comp.assign(_r=rng.random(len(comp))).sort_values("_r", kind="stable")
            .drop_duplicates("order_id", keep="last"))
    use_price = rng.random(len(pick)) < 0.5
    true = np.where(use_price, pick["price"], pick["freight_value"])
    tot = known[pick.index].values
    rec_sim = pay_brl.reindex(pick["order_id"]).values - (tot - true)
    err = np.abs(rec_sim - true)
    sim = {"orders_simulated": int(len(pick)),
           "exact_within_1_centavo": float((err <= 0.0101).mean()),
           "within_10_percent": float((err <= 0.1 * np.maximum(true, 1)).mean()),
           "median_abs_error_when_not_exact_brl": float(np.median(err[err > 0.0101]))}

    rec_val = pay_brl.reindex(it["order_id"]).values - known.values
    ok_order = (n_miss == 1).values & pay_complete.reindex(it["order_id"]).fillna(False).values
    it["amount_recovered"] = False
    for col, thr in [("price", PRICE_SCALE_THRESHOLD), ("freight_value", FREIGHT_SCALE_THRESHOLD)]:
        cand = ok_order & it[col].isna().values
        good = cand & (rec_val > 0) & (rec_val < thr)
        it.loc[good, col] = rec_val[good].round(2)
        it.loc[good, "amount_recovered"] = True
        it[f"{col}_recovered"] = good
        log("I3" if col == "price" else "I4", "items", col,
            "missing amount, recoverable from payments",
            f"tested on {sim['orders_simulated']:,} complete orders: exactly right "
            f"{sim['exact_within_1_centavo']:.1%} of the time (the rest are off by card interest)",
            "payment minus the other amounts in the order; flagged",
            good.sum(), it.loc[good, "order_id"])
        log("I3x" if col == "price" else "I4x", "items", col, "missing amount, not recoverable",
            "more than one missing in the order, payment missing, or result not believable",
            "kept missing; order total flagged incomplete", (cand & ~good).sum() +
            (it[col].isna() & ~cand).sum())
    sim["items_recovered"] = int(it["amount_recovered"].sum())
    return it, sim


def acknowledge(o: pd.DataFrame, it: pd.DataFrame, checks: dict) -> None:
    k = o.dropna(subset=["customer_unique_id", "order_purchase_timestamp"])
    dup = k[k.duplicated(["customer_unique_id", "order_purchase_timestamp"], keep=False)]
    sellers = it.groupby("order_id")["seller_id"].apply(lambda x: tuple(sorted(set(x))))
    products = it.groupby("order_id")["product_id"].apply(lambda x: tuple(sorted(x)))
    pay = o.set_index("order_id")["raw_payment_abs_centavos_sum"]
    same_seller = same_all = 0
    groups = dup.groupby(["customer_unique_id", "order_purchase_timestamp"])["order_id"]
    for _, ids in groups:
        ids = list(ids)
        same_seller += len({sellers.get(i) for i in ids}) == 1
        same_all += len({(products.get(i), pay.get(i)) for i in ids}) == 1
    n_groups = groups.ngroups
    log("A1", "orders", "customer_unique_id + order_purchase_timestamp",
        "orders sharing customer and exact purchase second",
        f"{n_groups} pairs; {same_seller} with the same seller; {n_groups - same_all} with different products or "
        f"payment (separate purchases); {same_all} with the same products and payment (possible duplicate)",
        f"kept; I can't prove they are duplicates, and {len(dup)} orders can't change the results", len(dup),
        dup["order_id"])
    cd = (o["order_status"] == "canceled") & o["order_delivered_customer_date"].notna()
    log("A2", "orders", "order_status + order_delivered_customer_date",
        "cancelled orders carrying a delivery date",
        "status says cancelled but there is a delivery date; the brief says cancelled orders get no label",
        "kept; no label because of the status", cd.sum())
    log("A3", "payments", "payment_type",
        "untrimmed / upper-case types and 'unclassified_payment'",
        f"{checks['payment_type_raw_variants']} raw variants -> {checks['payment_type_clean_variants']} after "
        f"trim+lower; {checks['payment_type_unclassified_rows']} unclassified rows",
        "left as is; payments are only used to check prices",
        checks["payment_type_unclassified_rows"])
    log("A4", "payments", "payment_installments", "negative installment counts (zero is valid per brief)",
        "Appendix B: zero installments can be valid; negatives cannot",
        "not used anywhere; noted only", checks["payment_installments_negative"])
    before_appr = (o["order_delivered_customer_date"].notna() & o["order_approved_at"].notna()
                   & (o["order_delivered_customer_date"] < o["order_approved_at"]))
    gap_h = ((o["order_approved_at"] - o["order_delivered_customer_date"]).dt.total_seconds() / 3600)[before_appr]
    o["delivered_before_approval"] = before_appr
    log("A6", "orders", "order_delivered_customer_date + order_approved_at",
        "delivered after purchase but before approval",
        f"approval saved {gap_h.min():.0f} to {gap_h.max():.0f} hours after delivery, so it looks like "
        "the approval was saved late",
        "kept and labelled; flagged as delivered_before_approval", before_appr.sum(), o.loc[before_appr, "order_id"])
    log("A5", "reviews", "review_score", "scores outside 1-5 (0 and above 5)",
        "valid scale is 1-5", "reviews are not loaded; just noted",
        checks["review_score_outside_1_5"])


def clean_orders(o: pd.DataFrame, zip3_map, zip_evidence, checks) -> pd.DataFrame:
    o = o.copy()
    st = o["order_status"].str.strip().str.lower()
    log("O1", "orders", "order_status", "untrimmed upper-case status values",
        "'  DELIVERED ' alongside 'delivered'", "strip + lower-case", (st != o["order_status"]).sum())
    o["order_status"] = st

    for c in TS_COLS:
        sent = o[c] == SENTINEL_TIMESTAMP
        log("T1", "orders", c, "placeholder timestamp '2018-02-30 10:00:00'",
            "impossible calendar date with a constant time", "set missing", sent.sum())
        parsed = pd.to_datetime(o[c].where(~sent), format=TS_FORMAT, errors="coerce")
        other_bad = o[c].notna() & ~sent & parsed.isna()
        log("T2", "orders", c, "other unparseable timestamps", "strict format parse",
            "set missing", other_bad.sum())
        o[c] = parsed
        o[c + "_was_sentinel"] = sent

    deliv, appr, purch = (o["order_delivered_customer_date"], o["order_approved_at"],
                          o["order_purchase_timestamp"])
    # fake deliveries: exactly 7 days before approval, to the second
    fab = (appr - deliv).dt.total_seconds() == FABRICATED_DELIVERY_OFFSET_S
    log("T3", "orders", "order_delivered_customer_date",
        "delivery exactly 7 days before approval (same clock time)",
        f"{int(fab.sum())} rows share this exact offset to the second; delivery would precede "
        "purchase, which can't happen", "set missing; order unlabelled", fab.sum())
    o["delivery_fabricated"] = fab
    o["order_delivered_customer_date"] = deliv.where(~fab)
    deliv = o["order_delivered_customer_date"]
    before = deliv.notna() & purch.notna() & (deliv < purch)
    log("T4", "orders", "order_delivered_customer_date", "other delivery before purchase",
        "impossible ordering", "order unlabelled", before.sum())
    o["delivery_before_purchase"] = before
    car = o["order_delivered_carrier_date"]
    cb = car.notna() & appr.notna() & (car < appr)
    log("T5", "orders", "order_delivered_carrier_date", "carrier handover before approval",
        "this also happens in the real Olist data; I don't use the carrier date for labels "
        "or features", "kept, flagged", cb.sum())
    o["carrier_before_approval"] = cb

    log("C1", "orders", "customer_id", "order's customer_id not in customers table",
        f"{checks['orders_missing_customer']} orders and {checks['customers_without_order']} customers unmatched; "
        f"{checks['orphan_orders_matching_after_trim_lower']} match after trimming and lower-casing, "
        f"{checks['orphan_orders_sharing_first_8_chars']} share the first 8 characters with an unmatched customer",
        "keep order; customer location unknown", (~o["customer_found"]).sum())
    z = o["customer_zip_code_prefix"]
    bz = z == SENTINEL_ZIP
    log("C2", "customers", "customer_zip_code_prefix", "placeholder zip '12A?0'",
        "non-numeric; real prefixes are 5 digits", "set missing", bz.sum())
    o["customer_zip_code_prefix"] = z.where(~bz)
    o["customer_state"], o["customer_state_source"] = clean_state(
        o["customer_state"], o["customer_zip_code_prefix"], zip3_map, "customers", "C3", o["customer_id"], zip_evidence)
    o.loc[~o["customer_found"], "customer_state_source"] = "customer_not_found"
    o["customer_city"] = o["customer_city"].str.strip().str.lower()
    return o


def label_orders(o: pd.DataFrame) -> pd.DataFrame:
    deliv = o["order_delivered_customer_date"]
    est = o["order_estimated_delivery_date"]
    status = o["order_status"]
    # first reason that matches wins
    conds = [
        (status.isin(["canceled", "unavailable"]), "cancelled_or_unavailable"),
        (status == "not_reported", "status_unknown"),
        (status != "delivered", "not_completed_status"),
        (o["delivery_fabricated"], "delivery_date_unreliable"),
        (o["order_delivered_customer_date_was_sentinel"], "delivery_date_unreliable"),
        (o["delivery_before_purchase"], "delivery_date_unreliable"),
        (deliv.isna(), "delivery_date_missing"),
        (est.isna(), "estimate_missing_or_unreliable"),
    ]
    reason = pd.Series("labelled", index=o.index)
    for cond, r in reversed(conds):
        reason = reason.mask(cond, r)
    o["label_status"] = reason
    ok = reason == "labelled"
    # compare calendar days: delivery on the estimated day counts as on time
    days = (deliv.dt.normalize() - est.dt.normalize()).dt.days
    o["delay_days_vs_estimate"] = days.where(ok)
    o["delivered_after_estimate"] = (days > 0).astype("Int8").where(ok)
    alt_ok = (reason == "status_unknown") & deliv.notna() & est.notna() & \
        ~o["delivery_fabricated"] & ~o["delivery_before_purchase"]
    o["label_if_status_unknown_included"] = (days > 0).astype("Int8").where(ok | alt_ok)
    # sensitivity only: cancelled orders that still have a valid delivery and estimate
    canc_ok = (status == "canceled") & deliv.notna() & est.notna() & \
        ~o["delivery_fabricated"] & ~o["delivery_before_purchase"]
    o["label_if_cancelled_included"] = (days > 0).astype("Int8").where(ok | canc_ok)
    return o


def aggregate_items(it: pd.DataFrame) -> pd.DataFrame:
    tmp = it.assign(_amount_missing=it.price.isna() | it.freight_value.isna(),
                    _repaired=it.price_repaired | it.freight_value_repaired,
                    _recovered=it.amount_recovered,
                    _prod_missing=~it.product_found, _seller_missing=~it.seller_found)
    g = tmp.groupby("order_id", sort=True)
    agg = g.agg(n_items=("order_item_id", "size"),
                n_distinct_products=("product_id", "nunique"),
                n_distinct_sellers=("seller_id", "nunique"),
                n_amount_missing=("_amount_missing", "sum"),
                any_amount_repaired=("_repaired", "any"),
                any_amount_recovered=("_recovered", "any"),
                n_product_missing=("_prod_missing", "sum"),
                n_seller_missing=("_seller_missing", "sum"))
    agg["items_price_sum"] = g["price"].sum(min_count=1)
    agg["items_freight_sum"] = g["freight_value"].sum(min_count=1)
    agg["amount_total_complete"] = agg["n_amount_missing"] == 0
    return agg


def _estimate() -> dict:
    m = json.loads((METRICS_DIR / "step1.json").read_text())["measured_in_memory_bytes"]
    inp = m["orders_raw"] + m["items_raw"]
    return {"inputs": inp, "peak_data_upper": 2 * inp}


def main() -> int:
    with step_metrics("step2") as rec:
        orders = pd.read_csv(IN / "orders_raw.csv", dtype={"customer_zip_code_prefix": "string",
                                                         "customer_state": "string",
                                                         "order_status": "string"})
        items = pd.read_csv(IN / "items_raw.csv", dtype={"seller_zip_code_prefix": "string",
                                                       "seller_state": "string",
                                                       "product_category_name": "string"})
        mem_in = df_bytes(orders) + df_bytes(items)
        pd.concat([profile(orders, "orders_raw"), profile(items, "items_raw")]).to_csv(
            OUT / "profile_before.csv", index=False)

        zip3_map, zip_stats = build_zip3_map([
            pd.DataFrame({"zip": orders["customer_zip_code_prefix"],
                          "state": orders["customer_state"].str.strip().str.upper()}),
            items.drop_duplicates("seller_id").pipe(lambda d: pd.DataFrame(
                {"zip": d["seller_zip_code_prefix"],
                 "state": d["seller_state"].str.strip().str.upper()}))])

        zip_evidence = (
            f"each zip range belongs to one state; {zip_stats['unambiguous_prefixes']} of "
            f"{zip_stats['zip3_prefixes']} 3-digit prefixes point to one state; built from a random half of "
            f"customers and sellers, it gets the other half right {zip_stats['holdout_accuracy']:.2%} of the "
            f"time ({zip_stats['holdout_covered_share']:.1%} covered); unclear prefixes stay unknown")
        evidence = {}
        it = clean_items(items, zip3_map, zip_evidence, evidence)
        it, recovery = recover_amounts(it, orders)
        checks = json.loads((IN / "load_checks.json").read_text())["relationship_checks"]
        o = clean_orders(orders, zip3_map, zip_evidence, checks)
        o = label_orders(o)
        acknowledge(o, it, checks)
        o = o.merge(aggregate_items(it), left_on="order_id", right_index=True, how="left",
                    validate="1:1")
        o["n_items"] = o["n_items"].fillna(0).astype("int16")
        o["has_items"] = o["n_items"] > 0
        o["any_amount_repaired"] = o["any_amount_repaired"].fillna(False).astype(bool)
        o["any_amount_recovered"] = o["any_amount_recovered"].fillna(False).astype(bool)
        o["payment_brl_abs"] = o["raw_payment_abs_centavos_sum"] / 100
        o["items_total_brl"] = o["items_price_sum"] + o["items_freight_sum"]
        log("PAY1", "payments", "payment_value", "centavos (100 = 1 BRL); negative and missing values",
            "brief Appendix B; negative ones match once the minus is removed",
            "divided by 100; only used to check prices",
            o["n_payment_negative"].sum())

        comp = o[o["amount_total_complete"].fillna(False) & (o["n_payment_null"] == 0) & o["has_items"]
                 & ~o["any_amount_recovered"]]
        match = (comp["payment_brl_abs"] - comp["items_total_brl"]).abs() < 0.015
        within_interest = ((comp["payment_brl_abs"] / comp["items_total_brl"]).between(0.999, 1.25))
        validation = {
            "zip3_state_map": zip_stats,
            "amount_rule_evidence": evidence,
            "amount_recovery_from_payments": recovery,
            "payments_vs_items": {
                "orders_compared": int(len(comp)),
                "exact_match_rate_repaired_orders": float(match[comp.any_amount_repaired].mean()),
                "exact_match_rate_untouched_orders": float(match[~comp.any_amount_repaired].mean()),
                "match_or_installment_interest_rate_repaired": float(within_interest[comp.any_amount_repaired].mean()),
                "match_or_installment_interest_rate_untouched": float(within_interest[~comp.any_amount_repaired].mean()),
                "n_repaired_orders_compared": int(comp.any_amount_repaired.sum()),
            },
        }

        outcomes = (o.groupby("label_status").size().rename("orders").reset_index()
                    .sort_values(["orders", "label_status"], ascending=[False, True], kind="stable"))
        outcomes["share"] = outcomes["orders"] / len(o)
        n_source = json.loads((IN / "load_checks.json").read_text())["relationship_checks"]["orders_rows"]
        assert outcomes["orders"].sum() == len(orders) == n_source
        assert o["order_id"].is_unique and len(it) == len(items)

        paths = {"orders_clean": OUT / "orders_clean.csv", "items_clean": OUT / "items_clean.csv"}
        o.to_csv(paths["orders_clean"], index=False, date_format=TS_FORMAT)
        it.to_csv(paths["items_clean"], index=False)
        outcomes.to_csv(OUT / "label_outcomes.csv", index=False)
        pd.concat([profile(o, "orders_clean"), profile(it, "items_clean")]).to_csv(
            OUT / "profile_after.csv", index=False)
        pd.DataFrame(LOG).to_csv(OUT / "cleaning_log.csv", index=False)
        write_json(OUT / "validation.json", validation)

        rec.update({
            "inputs_bytes": file_bytes([IN / "orders_raw.csv", IN / "items_raw.csv"]),
            "estimated_in_memory_bytes": _estimate(),
            "measured_in_memory_bytes": {"inputs": mem_in, "orders_clean": df_bytes(o),
                                         "items_clean": df_bytes(it)},
            "output_bytes": {p.name: file_bytes([p]) for p in OUT.iterdir()},
            "temporary_storage_bytes": 0,
        })
    print(outcomes.to_string(index=False))
    print(validation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
