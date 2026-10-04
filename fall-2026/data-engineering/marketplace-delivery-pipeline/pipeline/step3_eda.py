from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SNAPSHOT_DATE, STEP_DIRS, df_bytes, file_bytes, step_metrics, write_json

IN, OUT = STEP_DIRS["step2"], STEP_DIRS["step3"]
Z = 1.959964
SEED = 2040
MIN_N_STATE = 30
TAIL_START = "2018-08-01"
PERIODS = [("2016-01-01", "2017-11-01", "P1 to Oct-2017"),
           ("2017-11-01", "2018-05-01", "P2 Nov-2017 to Apr-2018 (spike)"),
           ("2018-05-01", TAIL_START, "P3 May to Jul-2018")]


# Wilson interval for a proportion (works for small groups)
def wilson(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return c - h, c + h


def rate_table(df, by, label="delivered_after_estimate"):
    g = df.groupby(by, observed=True)[label].agg(["sum", "count"]).rename(
        columns={"sum": "late", "count": "orders"})
    g["late_rate"] = g["late"] / g["orders"]
    g["ci_low"], g["ci_high"] = wilson(g["late"], g["orders"])
    return g.reset_index()


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def load():
    o = pd.read_csv(IN / "orders_clean.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date",
        "order_delivered_customer_date"], low_memory=False)
    o["window_days"] = (o["order_estimated_delivery_date"].dt.normalize()
                        - o["order_approved_at"].dt.normalize()).dt.days
    o["in_truncated_tail"] = o["order_estimated_delivery_date"] >= TAIL_START
    o["period"] = pd.NA
    for a, b, name in PERIODS:
        o.loc[(o.order_estimated_delivery_date >= a) & (o.order_estimated_delivery_date < b), "period"] = name
    return o


def q1(L, summary):
    k, n = int(L["delivered_after_estimate"].sum()), len(L)
    lo, hi = wilson(k, n)
    late = L[L["delivered_after_estimate"] == 1]["delay_days_vs_estimate"]
    summary["q1"] = {
        "labelled_orders": n, "late_orders": k, "late_rate": k / n,
        "late_rate_ci95": [float(lo), float(hi)],
        "late_delay_days": {"median": float(late.median()), "mean": float(late.mean()),
                            "p75": float(late.quantile(.75)), "p90": float(late.quantile(.9)),
                            "max": float(late.max())},
        "share_of_late_over_7_days": float((late > 7).mean()),
        "share_of_late_over_14_days": float((late > 14).mean()),
        "on_time_median_days_early": float(-L.loc[L.delivered_after_estimate == 0,
                                                   "delay_days_vs_estimate"].median()),
        "delivered_exactly_on_estimate_day": int((L["delay_days_vs_estimate"] == 0).sum()),
    }
    hist = L["delay_days_vs_estimate"].clip(-60, 60).value_counts().sort_index()
    hist.rename_axis("delay_days_clipped_pm60").rename("orders").to_csv(OUT / "q1_delay_distribution.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(hist.index, hist.values, width=1,
              color=np.where(hist.index > 0, "#c0392b", "#7f8c8d"))
    ax[0].axvline(0.5, color="k", lw=.8, ls="--")
    ax[0].set(title="Delivery date minus estimated date", xlabel="days (clipped at +/-60)",
              ylabel="orders")
    lb = late.clip(upper=60).value_counts().sort_index()
    ax[1].bar(lb.index, lb.values, width=1, color="#c0392b")
    ax[1].set(title=f"Late orders only (n={k:,}): days late", xlabel="days late (clipped at 60)",
              ylabel="orders")
    savefig(fig, "q1_delay_distribution.png")

    L = L.assign(month=L["order_estimated_delivery_date"].dt.to_period("M").astype(str))
    m = rate_table(L, "month")
    m["affected_by_extract_end"] = m["month"] >= TAIL_START[:7]
    m.to_csv(OUT / "q1_monthly_late_rate.csv", index=False)
    m = m[m.orders >= 50].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(m))
    ax.plot(x, m["late_rate"], marker="o", color="#c0392b")
    ax.fill_between(x, m["ci_low"], m["ci_high"], alpha=.2, color="#c0392b")
    tail = np.where(m["affected_by_extract_end"])[0]
    if len(tail):
        ax.axvspan(tail[0] - .5, x[-1] + .5, color="#7f8c8d", alpha=.18)
        ax.text(tail[0] - .6, ax.get_ylim()[1] * .92,
                f"extract ends {SNAPSHOT_DATE[:7]}:\nlate orders missing,\nrate biased down",
                fontsize=8, va="top", ha="right", color="#555")
    ax.set_xticks(x, m["month"], rotation=60)
    ax.set(title="Late-delivery rate by estimated-delivery month (months with >=50 orders, 95% CI)",
           ylabel="late rate")
    savefig(fig, "q1_monthly_late_rate.png")
    m = m[~m["affected_by_extract_end"]]
    summary["q1"]["monthly_rate_range"] = [float(m.late_rate.min()), float(m.late_rate.max())]
    summary["q1"]["worst_months"] = m.nlargest(3, "late_rate")[["month", "late_rate", "orders"]].to_dict("records")


def q2(L, summary):
    W = L.dropna(subset=["window_days"])
    bins = [-1, 10, 15, 20, 25, 30, 40, 1000]
    names = ["<=10", "11-15", "16-20", "21-25", "26-30", "31-40", ">40"]
    W = W.assign(window_bin=pd.cut(W["window_days"], bins, labels=names),
                 region=np.where(W["customer_state"] == "SP", "SP customers", "Other states"))
    t = rate_table(W, "window_bin")
    t.to_csv(OUT / "q2_window_vs_late.csv", index=False)
    tr = rate_table(W[W.customer_state.notna()], ["region", "window_bin"])
    tr.to_csv(OUT / "q2_window_vs_late_by_region.csv", index=False)
    rho = W[["window_days", "delivered_after_estimate"]].astype(float).corr(method="spearman").iloc[0, 1]
    late_w = W.groupby("delivered_after_estimate")["window_days"].median()
    tp = rate_table(W[W.period.notna()], ["period", "window_bin"])
    tp.to_csv(OUT / "q2_window_vs_late_by_period.csv", index=False)
    reg = window_regression(W[~W.in_truncated_tail & W.customer_state.notna()])
    summary["q2_adjusted"] = reg
    fig2, ax2 = plt.subplots(figsize=(7, 3.6))
    for (a, b, name), col in zip(PERIODS, ["#16a085", "#c0392b", "#2980b9"]):
        d = tp[(tp.period == name) & (tp.orders >= 30)]
        ax2.errorbar(d.window_bin.astype(str), d.late_rate,
                     yerr=[d.late_rate - d.ci_low, d.ci_high - d.late_rate], marker="o", capsize=3,
                     label=name, color=col)
    ax2.set(title="Late rate by promised window within time periods (bins >=30 orders)",
            xlabel="days from approval to estimated delivery", ylabel="late rate")
    ax2.legend(fontsize=8)
    savefig(fig2, "q2_window_vs_late_by_period.png")
    summary["q2"] = {
        "orders_with_window": len(W), "excluded_missing_approval": int(len(L) - len(W)),
        "window_days_median": float(W.window_days.median()),
        "window_days_p10_p90": [float(W.window_days.quantile(.1)), float(W.window_days.quantile(.9))],
        "median_window_late": float(late_w.get(1)), "median_window_on_time": float(late_w.get(0)),
        "spearman_window_vs_late": float(rho),
        "rate_by_bin": t[["window_bin", "orders", "late_rate", "ci_low", "ci_high"]].to_dict("records"),
    }
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(t))
    ax[0].bar(x, t.late_rate, color="#2c3e50",
              yerr=[t.late_rate - t.ci_low, t.ci_high - t.late_rate], capsize=3)
    for i, n in enumerate(t.orders):
        ax[0].text(i, 0.002, f"n={n:,}", ha="center", va="bottom", fontsize=7, color="white")
    ax[0].set(xticks=x, xticklabels=names, xlabel="days from approval to estimated delivery",
              ylabel="late rate", title="Late rate by promised window (95% CI)")
    for reg, col in [("SP customers", "#2980b9"), ("Other states", "#e67e22")]:
        d = tr[(tr.region == reg) & (tr.orders >= 30)]
        ax[1].errorbar(d.window_bin.astype(str), d.late_rate,
                       yerr=[d.late_rate - d.ci_low, d.ci_high - d.late_rate],
                       marker="o", capsize=3, label=reg, color=col)
    ax[1].set(title="Same, within SP vs other states (bins >=30 orders)", xlabel="window (days)",
              ylabel="late rate")
    ax[1].legend()
    savefig(fig, "q2_window_vs_late.png")


# logistic regression: window in weeks plus state and month dummies, bootstrap CI
def window_regression(W, reps=200):
    from sklearn.linear_model import LogisticRegression
    X = pd.concat([(W["window_days"] / 7).rename("window_weeks"),
                   pd.get_dummies(W["customer_state"], prefix="st", drop_first=True, dtype=float),
                   pd.get_dummies(W["order_estimated_delivery_date"].dt.to_period("M").astype(str),
                                  prefix="m", drop_first=True, dtype=float)], axis=1).values
    y = W["delivered_after_estimate"].astype(int).values

    def fit(Xb, yb):
        return LogisticRegression(C=1e6, solver="newton-cholesky", max_iter=100).fit(Xb, yb).coef_[0][0]

    b = fit(X, y)
    rng = np.random.default_rng(SEED)
    boots = [fit(X[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(reps))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"orders": int(len(y)), "controls": "customer state + estimated-delivery month",
            "odds_ratio_per_7_more_days": float(np.exp(b)),
            "ci95": [float(np.exp(lo)), float(np.exp(hi))], "bootstrap_reps": reps}


def q3(L, summary, rng):
    S = L[L["customer_state"].notna()]
    overall = S["delivered_after_estimate"].mean()
    t = rate_table(S, "customer_state").sort_values("customer_state", kind="stable").reset_index(drop=True)
    t["excess_late_orders"] = t["late"] - t["orders"] * overall
    t["share_zip_inferred"] = t["customer_state"].map(
        S.groupby("customer_state")["customer_state_source"].apply(lambda s: (s == "zip_inferred").mean()))
    t["median_window_days"] = t["customer_state"].map(S.groupby("customer_state")["window_days"].median())
    t["median_days_late_if_late"] = t["customer_state"].map(
        S[S.delivered_after_estimate == 1].groupby("customer_state")["delay_days_vs_estimate"].median())
    # each state's rate drawn from Beta(1 + late, 1 + on time), 4000 times
    draws = rng.beta(1 + t["late"].values[:, None], 1 + (t["orders"] - t["late"]).values[:, None],
                     size=(len(t), 4000))
    ranked = t["orders"].values >= MIN_N_STATE
    rd = np.where(ranked[:, None], draws, -1)
    top3 = np.argsort(-rd, axis=0, kind="stable")[:3]
    t["p_top3_rate"] = [(top3 == i).any(axis=0).mean() if ranked[i] else np.nan for i in range(len(t))]
    exc = (draws - overall) * t["orders"].values[:, None]
    top3e = np.argsort(-exc, axis=0, kind="stable")[:3]
    t["p_top3_excess"] = [(top3e == i).any(axis=0).mean() for i in range(len(t))]
    t["ranked"] = ranked
    t = t.sort_values(["late_rate", "customer_state"], ascending=[False, True], kind="stable").reset_index(drop=True)
    t.to_csv(OUT / "q3_state_late_rates.csv", index=False)
    NS = L[L["customer_state"].isna()]
    ns_lo, ns_hi = wilson(NS["delivered_after_estimate"].sum(), len(NS))
    summary["q3"] = {
        "orders_with_state": len(S), "orders_without_state": int(len(L) - len(S)),
        "late_rate_orders_without_state": float(NS["delivered_after_estimate"].mean()),
        "late_rate_orders_without_state_ci95": [float(ns_lo), float(ns_hi)],
        "overall_rate_with_state": float(overall),
        "states": int(len(t)), "states_below_min_n": t.loc[~t.ranked, "customer_state"].tolist(),
        "top_rate_states": t[t.ranked].head(6)[["customer_state", "orders", "late_rate", "ci_low",
                                                 "ci_high", "p_top3_rate"]].to_dict("records"),
        "top_excess_states": t.sort_values(["excess_late_orders", "customer_state"], ascending=[False, True], kind="stable").head(6)[
            ["customer_state", "orders", "late", "late_rate", "excess_late_orders",
             "p_top3_excess"]].to_dict("records"),
        "lowest_rate_states": t[t.ranked].tail(4)[["customer_state", "orders", "late_rate"]].to_dict("records"),
    }
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    tt = t.iloc[::-1]
    col = np.where(tt.ranked, "#c0392b", "#bdc3c7")
    y = np.arange(len(tt))
    ax[0].barh(y, tt.late_rate, color=col, xerr=[tt.late_rate - tt.ci_low, tt.ci_high - tt.late_rate],
               capsize=2, error_kw={"lw": .8})
    ax[0].axvline(overall, color="k", ls="--", lw=.8)
    ax[0].set(yticks=y, yticklabels=[f"{s} (n={n:,})" for s, n in zip(tt.customer_state, tt.orders)],
              xlabel="late rate (95% Wilson CI)",
              title=f"Late rate by customer state (grey: n<{MIN_N_STATE})")
    ax[0].tick_params(axis="y", labelsize=7)
    e = t.sort_values(["excess_late_orders", "customer_state"], ascending=[True, False], kind="stable").tail(12)
    ax[1].barh(e.customer_state, e.excess_late_orders, color="#8e44ad")
    ax[1].set(xlabel="late orders above the national rate",
              title="Where the excess late orders are (top 12)")
    savefig(fig, "q3_state_late_rates.png")
    return t


def headline(df, label="delivered_after_estimate"):
    d = df.dropna(subset=[label])
    S = d[d.customer_state.notna()]
    t = S.groupby("customer_state")[label].agg(["mean", "count"])
    t = (t[t["count"] >= MIN_N_STATE].reset_index().sort_values(["mean", "customer_state"], ascending=[False, True],
                                                               kind="stable").set_index("customer_state"))
    ov = S[label].mean()
    ex = (S.groupby("customer_state")[label].sum() - S.groupby("customer_state").size() * ov)
    W = d.dropna(subset=["window_days"])
    short = W[W.window_days <= 15][label].mean()
    long_ = W[W.window_days > 30][label].mean()
    return {"orders": len(d), "late_rate": float(d[label].mean()),
            "late_rate_window_le15": float(short), "late_rate_window_gt30": float(long_),
            "top3_rate_states": ",".join(t.index[:3]),
            "top3_excess_states": ",".join(ex.rename("ex").reset_index().sort_values(["ex", "customer_state"], ascending=[False, True], kind="stable")["customer_state"].head(3)),
            "state_rates": t["mean"]}


# redo the headline numbers with different rules for which orders count
def sensitivity(o, summary):
    base = o[o.label_status == "labelled"]
    alt = o.assign(delivered_after_estimate=o["label_if_status_unknown_included"])
    scen = {
        "S0 main (labelled orders)": base,
        "S1 + status 'not_reported' with valid delivery data": alt,
        "S2 only orders with original (not zip-inferred) state": base[base.customer_state_source == "original"],
        "S3 exclude orders with repaired or missing amounts":
            base[~base.any_amount_repaired & base.amount_total_complete.fillna(False)],
        "S4 exclude orders with any product/seller not found":
            base[(base.n_product_missing == 0) & (base.n_seller_missing == 0)],
        "S5 exclude orders with carrier handover before approval": base[~base.carrier_before_approval],
        "S6 exclude estimates from Aug-2018 (extract end)": base[~base.in_truncated_tail],
        f"S7 exclude estimates after {SNAPSHOT_DATE}": base[base.order_estimated_delivery_date <= SNAPSHOT_DATE],
        "S8 exclude orders with an amount recovered from payments": base[~base.any_amount_recovered],
        "S9 + cancelled orders that have a valid delivery date":
            o.assign(delivered_after_estimate=o["label_if_cancelled_included"]),
    }
    rows, ref = [], None
    for name, d in scen.items():
        h = headline(d)
        sr = h.pop("state_rates")
        if ref is None:
            ref = sr
        common = ref.index.intersection(sr.index)
        h["spearman_state_rates_vs_S0"] = float(ref[common].rank().corr(sr[common].rank()))
        rows.append({"scenario": name, **h})
    unk = o[(o.order_status == "delivered") &
            o.label_status.isin(["delivery_date_unreliable", "delivery_date_missing",
                                 "estimate_missing_or_unreliable"])]
    n, k, u = len(base), base.delivered_after_estimate.sum(), len(unk)
    for name, extra in [("B1 bound: all unmeasurable delivered orders on time", 0),
                        ("B2 bound: all unmeasurable delivered orders late", u)]:
        rows.append({"scenario": name, "orders": n + u, "late_rate": float((k + extra) / (n + u))})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "sensitivity.csv", index=False)
    summary["sensitivity"] = s.to_dict("records")
    summary["unmeasurable_delivered_orders"] = int(u)


def main() -> int:
    with step_metrics("step3") as rec:
        rng = np.random.default_rng(SEED)
        o = load()
        L = o[o.label_status == "labelled"].copy()
        summary = {}
        q1(L, summary)
        q2(L, summary)
        q3(L, summary, rng)
        sensitivity(o, summary)
        write_json(OUT / "summary.json", summary)
        rec.update({
            "inputs_bytes": file_bytes([IN / "orders_clean.csv"]),
            "estimated_in_memory_bytes": {
                "orders_clean": json.loads((STEP_DIRS["step2"].parent / "metrics" / "step2.json")
                                                         .read_text())["measured_in_memory_bytes"]["orders_clean"],
                "posterior_draws": 27 * 4000 * 8 * 3},
            "measured_in_memory_bytes": {"orders_clean": df_bytes(o)},
            "output_bytes": {p.name: file_bytes([p]) for p in OUT.iterdir()},
            "temporary_storage_bytes": 0,
        })
    print(pd.DataFrame(summary["sensitivity"]).drop(columns=[], errors="ignore").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
