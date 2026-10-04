from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import BRAZIL_STATES, STEP_DIRS, df_bytes, file_bytes, step_metrics, write_json

IN, OUT = STEP_DIRS["step2"], STEP_DIRS["step4"]
# columns that are not known at approval (from the brief)
BANNED_SOURCES = {"payment_value", "payment_type", "payment_installments", "payment_sequential",
                  "order_status", "order_delivered_carrier_date", "order_delivered_customer_date",
                  "review_score", "review_comment_title", "review_comment_message",
                  "review_creation_date", "review_answer_timestamp", "shipping_limit_date",
                  "delay_days_vs_estimate", "delivered_after_estimate", "label_status"}
TOP_SELLER_STATES = ["SP", "PR", "MG", "RJ", "SC"]
MIN_CATEGORY_SHARE = 0.01
DICT: list[dict] = []


def add(F, name, values, meaning, sources, transform):
    F[name] = values
    DICT.append({"feature": name, "meaning": meaning, "source_columns": sources,
                 "transformation": transform, "available_at_approval": True})


# hour, weekday and month as sin/cos so the end wraps around to the start
def cyc(F, prefix, values, period, meaning, sources):
    ang = 2 * np.pi * values / period
    add(F, f"{prefix}_sin", np.sin(ang), meaning + " (cyclical sine)", sources, f"sin(2*pi*x/{period})")
    add(F, f"{prefix}_cos", np.cos(ang), meaning + " (cyclical cosine)", sources, f"cos(2*pi*x/{period})")


# quick leakage test: a leaked feature would give AUC close to 1
def sanity_auc(F, y, when, tail):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    m = HistGradientBoostingClassifier(random_state=0)
    cv = cross_val_score(m, F, y, cv=5, scoring="roc_auc")
    order = np.argsort(when.values, kind="stable")
    cut = int(len(order) * 0.8)
    tr, te = order[:cut], order[cut:]
    m.fit(F.iloc[tr], y.iloc[tr])
    keep = np.where(~tail)[0]
    order2 = keep[np.argsort(when.values[keep], kind="stable")]
    cut2 = int(len(order2) * 0.8)
    m2 = HistGradientBoostingClassifier(random_state=0).fit(F.iloc[order2[:cut2]], y.iloc[order2[:cut2]])
    auc2 = roc_auc_score(y.iloc[order2[cut2:]], m2.predict_proba(F.iloc[order2[cut2:]])[:, 1])
    return {"model": "HistGradientBoostingClassifier (defaults)",
            "time_split_excl_extract_end_roc_auc_last20pct": round(float(auc2), 4),
            "late_rate_test_excl_extract_end": round(float(y.iloc[order2[cut2:]].mean()), 4),
            "cv5_roc_auc_mean": round(float(cv.mean()), 4), "cv5_roc_auc_sd": round(float(cv.std()), 4),
            "time_split_roc_auc_last20pct": round(float(roc_auc_score(y.iloc[te], m.predict_proba(F.iloc[te])[:, 1])), 4),
            "late_rate_train": round(float(y.iloc[tr].mean()), 4), "late_rate_test": round(float(y.iloc[te].mean()), 4)}


def main() -> int:
    with step_metrics("step4") as rec:
        o = pd.read_csv(IN / "orders_clean.csv", low_memory=False, parse_dates=[
            "order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date"])
        it = pd.read_csv(IN / "items_clean.csv", low_memory=False)

        hist = o.dropna(subset=["customer_unique_id", "order_purchase_timestamp"]).sort_values(
            "order_purchase_timestamp")
        hist["prior"] = hist.groupby("customer_unique_id")["order_purchase_timestamp"].rank(
            method="min") - 1
        o["prior_orders_customer"] = o["order_id"].map(hist.set_index("order_id")["prior"])

        R = o[o["label_status"] == "labelled"].reset_index(drop=True)
        n = len(R)
        imput = {}

        it = it[it["order_id"].isin(R["order_id"])].copy()
        # prices rebuilt from payments count as missing here, payments are not allowed as predictors
        for col in ["price", "freight_value"]:
            it.loc[it[f"{col}_recovered"].astype(bool), col] = np.nan
        it["main_rank"] = it.groupby("order_id")["price"].rank(method="first", ascending=False)
        g = it.groupby("order_id")
        A = pd.DataFrame({
            "max_price": g["price"].max(),
            "price_sum": g["price"].sum(min_count=1),
            "freight_sum": g["freight_value"].sum(min_count=1),
            "amount_missing": (it["price"].isna() | it["freight_value"].isna()).groupby(it["order_id"]).any(),
            "weight_sum": g["product_weight_g"].sum(min_count=1),
            "volume_sum": g["volume_cm3"].sum(min_count=1),
            "photos_mean": g["product_photos_qty"].mean(),
            "desc_len_mean": g["product_description_length"].mean(),
            "name_len_mean": g["product_name_length"].mean(),
            "share_product_missing": 1 - g["product_found"].mean(),
            "share_seller_missing": 1 - g["seller_found"].mean(),
            "n_seller_states": g["seller_state"].nunique(),
        })
        same = it.merge(R[["order_id", "customer_state"]], on="order_id")
        same["same"] = same["seller_state"] == same["customer_state"]
        A["share_items_same_state"] = same.groupby("order_id")["same"].mean()
        main = it[it["main_rank"] == 1].set_index("order_id")
        A["main_category"] = main["category"]
        A["main_seller_state"] = main["seller_state"]
        for st in TOP_SELLER_STATES:
            A[f"share_seller_{st}"] = (it["seller_state"] == st).groupby(it["order_id"]).mean()
        R = R.merge(A, left_on="order_id", right_index=True, how="left", validate="1:1")

        F = pd.DataFrame(index=R.index)
        appr = R["order_approved_at"]
        lag_h = (appr - R["order_purchase_timestamp"]).dt.total_seconds() / 3600
        med_lag = float(lag_h.median())
        imput["purchase_to_approval_hours_median"] = med_lag
        appr_filled = appr.fillna(R["order_purchase_timestamp"] + pd.to_timedelta(med_lag, "h"))
        add(F, "approval_time_missing", appr.isna().astype(int),
            "1 if approval timestamp was missing/placeholder and was estimated", "order_approved_at",
            "indicator")
        window = (R["order_estimated_delivery_date"].dt.normalize() - appr_filled.dt.normalize()).dt.days
        imput["window_days_median"] = float(window.median())
        add(F, "window_days", window.fillna(window.median()),
            "calendar days from approval to the estimated delivery date (same as in step 3)",
            "order_approved_at, order_estimated_delivery_date", "calendar days; median fill")
        add(F, "purchase_to_approval_hours", lag_h.fillna(med_lag).clip(upper=24 * 30),
            "hours between purchase and approval", "order_purchase_timestamp, order_approved_at",
            "difference in hours; clipped at 720; median fill")
        cyc(F, "approval_hour", appr_filled.dt.hour.fillna(12), 24, "hour of day of approval",
            "order_approved_at")
        cyc(F, "approval_dow", appr_filled.dt.dayofweek.fillna(0), 7, "weekday of approval",
            "order_approved_at")
        cyc(F, "approval_month", appr_filled.dt.month.fillna(6) - 1, 12, "month of approval (season)",
            "order_approved_at")
        cyc(F, "estimate_dow", R["order_estimated_delivery_date"].dt.dayofweek, 7,
            "weekday of the estimated delivery date", "order_estimated_delivery_date")

        for col, name, meaning in [("n_items", "n_items", "number of items"),
                                   ("n_distinct_products", "n_distinct_products", "distinct products"),
                                   ("n_distinct_sellers", "n_distinct_sellers", "distinct sellers")]:
            add(F, name, R[col].astype(float), meaning + " in the order", "order_items", "count")
        price, freight = R["price_sum"], R["freight_sum"]
        for v, name, meaning in [(price, "log_price_total", "total item price (BRL)"),
                                 (freight, "log_freight_total", "total freight (BRL)"),
                                 (R["max_price"], "log_max_item_price", "most expensive item (BRL)")]:
            med = float(v.median()); imput[name] = med
            add(F, name, np.log1p(v.fillna(med)), meaning, "order_items.price/freight_value (cleaned)",
                "log1p; median fill")
        add(F, "freight_to_price_ratio", (freight / price.replace(0, np.nan)).fillna(
            (freight / price).median()).clip(upper=5),
            "freight relative to item price", "order_items.price, freight_value", "ratio; clip 5; median fill")
        add(F, "amount_missing", R["amount_missing"].fillna(True).astype(int),
            "1 if any item price or freight is missing (amounts rebuilt from payments count as missing, "
            "because payments are not allowed as predictors)", "order_items", "indicator")
        for col, name, meaning in [("weight_sum", "log_weight_total_g", "total product weight (g)"),
                                   ("volume_sum", "log_volume_total_cm3", "total product volume (cm3)")]:
            med = float(R[col].median()); imput[name] = med
            add(F, name, np.log1p(R[col].fillna(med)), meaning, "products (cleaned)", "log1p; median fill")
        for col, name, meaning in [("photos_mean", "photos_mean", "mean product photos"),
                                   ("desc_len_mean", "description_length_mean", "mean description length"),
                                   ("name_len_mean", "name_length_mean", "mean product-name length")]:
            med = float(R[col].median()); imput[name] = med
            add(F, name, R[col].fillna(med), meaning, "products (optional fields)", "mean; median fill")
        add(F, "product_attrs_missing", (R["weight_sum"].isna() | R["photos_mean"].isna()).astype(int),
            "1 if product size or catalogue attributes are missing", "products", "indicator")

        cat_share = R["main_category"].value_counts(normalize=True)
        keep = sorted(cat_share[cat_share >= MIN_CATEGORY_SHARE].index)
        for c in keep:
            add(F, f"cat_{c}", (R["main_category"] == c).astype(int),
                f"main (highest-price) item in category '{c}'", "products.product_category_name",
                "one-hot (categories >=1% of orders); reference = rarer categories")
        add(F, "cat_missing", R["main_category"].isna().astype(int),
            "main item category unknown", "products.product_category_name", "indicator")

        cs = R["customer_state"]
        for st in sorted(BRAZIL_STATES):
            if st == "SP":
                continue
            add(F, f"cust_state_{st}", (cs == st).astype(int), f"customer in state {st}",
                "customers.customer_state (cleaned)", "one-hot; reference = SP")
        add(F, "cust_state_unknown", cs.isna().astype(int), "customer state unknown",
            "customers", "indicator")
        add(F, "share_items_same_state", R["share_items_same_state"].fillna(0),
            "share of items whose seller is in the customer's state", "sellers, customers",
            "share; 0 if unknown")
        for st in TOP_SELLER_STATES:
            add(F, f"share_seller_{st}", R[f"share_seller_{st}"].fillna(0),
                f"share of items shipped by sellers in {st}", "sellers.seller_state (cleaned)", "share")
        add(F, "n_seller_states", R["n_seller_states"].fillna(0),
            "number of distinct seller states in the order", "sellers", "count")
        add(F, "share_seller_not_found", R["share_seller_missing"],
            "share of items whose seller is absent from the sellers table", "order_items, sellers", "share")
        add(F, "customer_history_unknown", R["prior_orders_customer"].isna().astype(int),
            "1 if the customer record is missing, so prior orders cannot be counted",
            "customers.customer_unique_id", "indicator")
        add(F, "prior_orders_customer", R["prior_orders_customer"].fillna(0),
            "earlier orders by the same customer (customer_unique_id) in this extract",
            "customers.customer_unique_id, order_purchase_timestamp",
            "count of strictly earlier purchases; 0 when unknown (see customer_history_unknown)")

        F = F.astype(float)
        ids = R[["order_id", "delivered_after_estimate"]].astype({"delivered_after_estimate": int})

        dct = pd.DataFrame(DICT)
        used = {s.strip().split(".")[-1] for src in dct["source_columns"] for s in src.split(",")}
        corr = F.corrwith(ids["delivered_after_estimate"].astype(float)).abs().sort_values(ascending=False)
        checks = {
            "rows": n, "features": F.shape[1],
            "rows_equal_retained_orders": n == int((o.label_status == "labelled").sum()),
            "order_ids_unique": bool(ids.order_id.is_unique),
            "all_numeric": bool(all(k.kind == "f" for k in F.dtypes)),
            "no_missing_values": bool(F.notna().all().all()),
            "no_infinite_values": bool(np.isfinite(F.values).all()),
            "labels_binary": sorted(ids.delivered_after_estimate.unique().tolist()),
            "banned_source_columns_used": sorted(used & BANNED_SOURCES),
            "every_feature_documented": bool(set(F.columns) == set(dct.feature)),
            "max_abs_corr_with_label": corr.head(8).round(4).to_dict(),
            "leakage_flag_corr_over_0.5": corr[corr > .5].index.tolist(),
            "constant_features": F.columns[F.nunique() <= 1].tolist(),
            "full_column_rank_with_intercept": bool(np.linalg.matrix_rank(
                np.column_stack([np.ones(len(F)), F.values])) == F.shape[1] + 1),
            "calendar_fields_all_cyclical": all(
                c.endswith(("_sin", "_cos")) for c in F.columns
                if any(t in c for t in ("_dow", "_hour_", "_month_"))),
            "fitted_on_all_retained_rows": f"imputation medians and the kept-category list use all "
                                           f"{n:,} rows (no labels); for modelling, refit them on the "
                                           "training fold only",
        }
        tail = (R["order_estimated_delivery_date"] >= "2018-08-01").values
        checks["leakage_sanity_model"] = sanity_auc(F, ids["delivered_after_estimate"], appr_filled, tail)
        assert not checks["banned_source_columns_used"] and checks["no_missing_values"]
        assert checks["rows_equal_retained_orders"] and checks["every_feature_documented"]

        F.to_csv(OUT / "features.csv", index=False, float_format="%.6g")
        ids.to_csv(OUT / "order_ids_labels.csv", index=False)
        dct.to_csv(OUT / "feature_dictionary.csv", index=False)
        imput["note"] = ("Fitted on all retained orders without using labels. In a modelling pipeline, "
                         "refit these on the training fold only.")
        write_json(OUT / "imputation_values.json", imput)
        write_json(OUT / "feature_checks.json", checks)
        rec.update({
            "inputs_bytes": file_bytes([IN / "orders_clean.csv", IN / "items_clean.csv"]),
            "estimated_in_memory_bytes": {"features": n * F.shape[1] * 8},
            "measured_in_memory_bytes": {"features": df_bytes(F), "orders_clean": df_bytes(o)},
            "output_bytes": {p.name: file_bytes([p]) for p in OUT.iterdir()},
            "temporary_storage_bytes": 0,
        })
    print(f"step4: {F.shape[0]:,} rows x {F.shape[1]} features")
    print(checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
