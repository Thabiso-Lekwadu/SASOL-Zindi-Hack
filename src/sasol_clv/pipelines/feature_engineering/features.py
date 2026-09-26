# customer level features built only from rows before the cutoff
import warnings

import numpy as np
import pandas as pd

# we add many columns one by one, pandas complains but it is fine at this size
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

RHYTHM_CATEGORIES = [
    "Fuel", "Beverages", "Food & beverage", "Confectionary", "Snacking", "Tobacco", "Breads & rolls",
    "Groceries", "Lubricants", "Ice-cream", "Braai & Ice", "Coffee", "Seattle", "Perishables",
]
GROWTH_CATEGORIES = [
    "Fuel", "Beverages", "Food & beverage", "Confectionary", "Snacking", "Tobacco", "Breads & rolls",
    "Groceries", "Lubricants",
]


def _safe_name(cat: str) -> str:
    return cat.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")


def build_features(tx: pd.DataFrame, cutoff, customers, windows, focus_categories) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    customers = pd.Index(customers, name="ID")
    hist = tx[(tx["DateTimeZA"] < cutoff) & tx["ID"].isin(customers)].copy()
    hist["days_ago"] = (cutoff - hist["DateTimeZA"]).dt.total_seconds() / 86400.0

    feats = pd.DataFrame(index=customers)

    # basket level view, one row per basket
    baskets = hist.groupby(["ID", "TransactionId"]).agg(
        ts=("DateTimeZA", "min"),
        days_ago=("days_ago", "min"),
        total=("TransactionTotalInclAmount", "first"),
        site=("LocationId", "first"),
    ).reset_index()

    g = baskets.groupby("ID")
    feats["n_baskets"] = g.size()
    feats["recency_days"] = g["days_ago"].min()
    feats["tenure_days"] = g["days_ago"].max()
    feats["n_sites"] = g["site"].nunique()
    feats["avg_basket_total"] = g["total"].mean()
    feats["active_days"] = baskets.assign(d=baskets["ts"].dt.date).groupby("ID")["d"].nunique()
    feats["baskets_per_month"] = feats["n_baskets"] / (feats["tenure_days"].clip(lower=1) / 30.4)

    # gaps between visits say a lot about churn
    gaps = baskets.sort_values(["ID", "ts"]).groupby("ID")["ts"].diff().dt.total_seconds() / 86400.0
    baskets["gap"] = gaps
    gg = baskets.groupby("ID")["gap"]
    feats["gap_mean"] = gg.mean()
    feats["gap_median"] = gg.median()
    feats["gap_std"] = gg.std()
    feats["recency_over_gap"] = feats["recency_days"] / feats["gap_mean"].clip(lower=0.5)

    fuel_rows = hist[hist["is_fuel_litres"]]
    nonfuel_rows = hist[hist["is_nonfuel"]]

    # lifetime totals
    feats["fuel_litres_all"] = fuel_rows.groupby("ID")["Quantity"].sum()
    feats["fuel_rand_all"] = fuel_rows.groupby("ID")["TotalExclAmount"].sum()
    feats["nonfuel_all"] = nonfuel_rows.groupby("ID")["TotalExclAmount"].sum()
    months_seen = (feats["tenure_days"].clip(lower=30) / 30.4)
    feats["fuel_litres_per_month"] = feats["fuel_litres_all"] / months_seen
    feats["nonfuel_per_month"] = feats["nonfuel_all"] / months_seen

    # recent windows
    for w in windows:
        fw = fuel_rows[fuel_rows["days_ago"] <= w]
        nw = nonfuel_rows[nonfuel_rows["days_ago"] <= w]
        bw = baskets[baskets["days_ago"] <= w]
        feats[f"fuel_litres_{w}d"] = fw.groupby("ID")["Quantity"].sum()
        feats[f"fuel_baskets_{w}d"] = fw.groupby("ID")["TransactionId"].nunique()
        feats[f"nonfuel_{w}d"] = nw.groupby("ID")["TotalExclAmount"].sum()
        feats[f"nonfuel_baskets_{w}d"] = nw.groupby("ID")["TransactionId"].nunique()
        feats[f"baskets_{w}d"] = bw.groupby("ID").size()

    # the quarter right before the cutoff versus the one before that, is the customer speeding up or slowing down
    q1 = hist[hist["days_ago"] <= 91]
    q2 = hist[(hist["days_ago"] > 91) & (hist["days_ago"] <= 182)]
    for name, part in [("q1", q1), ("q2", q2)]:
        feats[f"fuel_litres_{name}"] = part[part["is_fuel_litres"]].groupby("ID")["Quantity"].sum()
        feats[f"nonfuel_{name}"] = part[part["is_nonfuel"]].groupby("ID")["TotalExclAmount"].sum()
    feats = feats.fillna({c: 0 for c in feats.columns if c.startswith(("fuel_", "nonfuel_", "baskets_"))})
    feats["fuel_trend"] = np.log1p(feats["fuel_litres_q1"].clip(lower=0)) - np.log1p(feats["fuel_litres_q2"].clip(lower=0))
    feats["nonfuel_trend"] = np.log1p(feats["nonfuel_q1"].clip(lower=0)) - np.log1p(feats["nonfuel_q2"].clip(lower=0))

    # log versions of the recent totals, same scale as the target
    feats["log_fuel_q1"] = np.log1p(feats["fuel_litres_q1"].clip(lower=0))
    feats["log_nonfuel_q1"] = np.log1p(feats["nonfuel_q1"].clip(lower=0))

    # share of the wallet going to fuel
    feats["fuel_share"] = feats["fuel_rand_all"] / (feats["fuel_rand_all"] + feats["nonfuel_all"].clip(lower=0)).replace(0, np.nan)

    # category features, these drive the opportunity label
    inscope = hist[hist["in_scope"]]
    qual = hist[hist["is_qualifying"]]
    feats["n_categories_ever"] = qual.groupby("ID")["ItemCategoryLevel2"].nunique()
    feats["n_categories_q1"] = qual[qual["days_ago"] <= 91].groupby("ID")["ItemCategoryLevel2"].nunique()

    for cat in focus_categories:
        key = _safe_name(cat)
        c_all = inscope[inscope["ItemCategoryLevel2"] == cat]
        feats[f"cat_{key}_ever"] = qual[qual["ItemCategoryLevel2"] == cat].groupby("ID").size().gt(0).astype(float)
        feats[f"cat_{key}_rand_q1"] = c_all[c_all["days_ago"] <= 91].groupby("ID")["TotalExclAmount"].sum()
        feats[f"cat_{key}_rand_q2"] = c_all[(c_all["days_ago"] > 91) & (c_all["days_ago"] <= 182)].groupby("ID")["TotalExclAmount"].sum()
        feats[f"cat_{key}_baskets_all"] = c_all.groupby("ID")["TransactionId"].nunique()

    cat_cols = [c for c in feats.columns if c.startswith(("cat_", "n_categories"))]
    feats[cat_cols] = feats[cat_cols].fillna(0)

    # we tried a cutoff month feature, it made validation worse so it is left out

    # extra columns start with x_, only the opportunity classifier uses them
    extra = pd.concat(
        [rhythm_features(hist, cutoff, customers, tx), growth_features(hist, cutoff, customers)], axis=1
    )
    feats = feats.join(extra)

    return feats.reset_index().copy()


def rhythm_features(hist: pd.DataFrame, cutoff, customers, tx: pd.DataFrame) -> pd.DataFrame:
    # how recently and how regularly the customer shows up, per category and overall
    f = pd.DataFrame(index=customers)
    qual = hist[hist["is_qualifying"]]

    last = qual.groupby(["ID", "ItemCategoryLevel2"])["days_ago"].min().unstack()
    for cat in RHYTHM_CATEGORIES:
        f[f"x_{_safe_name(cat)}_days_since"] = last[cat] if cat in last else np.nan

    all_cats = tx.loc[tx["in_scope"], "ItemCategoryLevel2"].nunique()
    f["x_n_cats_never"] = all_cats - qual.groupby("ID")["ItemCategoryLevel2"].nunique().reindex(customers).fillna(0)

    # weeks active out of the last 13, and last 4 weeks versus the 9 before
    baskets = hist.drop_duplicates("TransactionId").copy()
    baskets["week"] = (baskets["days_ago"] // 7).astype(int)
    f["x_weeks_active_13"] = baskets[baskets["week"] < 13].groupby("ID")["week"].nunique()
    f["x_baskets_4w"] = baskets[baskets["week"] < 4].groupby("ID").size()
    f["x_baskets_5_13w"] = baskets[(baskets["week"] >= 4) & (baskets["week"] < 13)].groupby("ID").size()
    cols = ["x_weeks_active_13", "x_baskets_4w", "x_baskets_5_13w"]
    f[cols] = f[cols].fillna(0)
    f["x_basket_rate_ratio"] = (f["x_baskets_4w"] / 4 + 0.1) / (f["x_baskets_5_13w"] / 9 + 0.1)

    fuel = hist[hist["is_fuel_litres"]]
    f["x_fuel_litres_4w"] = fuel[fuel["days_ago"] < 28].groupby("ID")["Quantity"].sum()
    f["x_fuel_litres_4w"] = f["x_fuel_litres_4w"].fillna(0)
    f["x_fuel_litres_per_fill"] = fuel.groupby(["ID", "TransactionId"])["Quantity"].sum().groupby("ID").median()
    is_diesel = fuel["ItemCategoryLevel3"].str.contains("Diesel", case=False)
    f["x_diesel_share"] = (is_diesel * fuel["Quantity"]).groupby(fuel["ID"]).sum() / fuel.groupby("ID")[
        "Quantity"
    ].sum().replace(0, np.nan)

    # slope of monthly spend over the last 6 months, positive means speeding up
    month = (hist["days_ago"] // 30.4).astype(int)
    steps = np.arange(6) - 2.5
    for name, mask, col in [
        ("fuel", hist["is_fuel_litres"], "Quantity"),
        ("nonfuel", hist["is_nonfuel"], "TotalExclAmount"),
    ]:
        part = hist[mask & (month < 6)].assign(month=month)
        monthly = (
            part.groupby(["ID", "month"])[col]
            .sum()
            .unstack()
            .reindex(index=customers, columns=range(6))
            .fillna(0)
        )
        monthly = np.log1p(monthly.clip(lower=0))
        f[f"x_{name}_slope6"] = (monthly.values * (-steps)).sum(1) / (steps**2).sum()
        f[f"x_{name}_months_active6"] = (monthly > 0).sum(1)

    f["x_weekend_share"] = baskets["DateTimeZA"].dt.dayofweek.ge(5).groupby(baskets["ID"]).mean()
    f["x_returns_count"] = hist[(hist["Quantity"] < 0) | (hist["TotalExclAmount"] < 0)].groupby("ID").size()
    f["x_returns_count"] = f["x_returns_count"].fillna(0)
    return f


def growth_features(hist: pd.DataFrame, cutoff, customers) -> pd.DataFrame:
    # compares last quarter spend per category with the customer's usual quarter
    # and with the rand amount they would need next quarter to count as growth
    f = pd.DataFrame(index=customers)
    inscope = hist[hist["in_scope"]]
    prev_start = cutoff - pd.DateOffset(months=3)

    first_seen = hist.groupby("ID")["DateTimeZA"].min().reindex(customers)
    quarters_seen = ((cutoff - first_seen).dt.days / 91.3).clip(lower=1)

    total = inscope.groupby(["ID", "ItemCategoryLevel2"])["TotalExclAmount"].sum().unstack()
    prev = (
        inscope[inscope["DateTimeZA"] >= prev_start]
        .groupby(["ID", "ItemCategoryLevel2"])["TotalExclAmount"]
        .sum()
        .unstack()
    )

    def column(table, cat):
        if cat in table:
            return table[cat].reindex(customers).fillna(0)
        return pd.Series(0.0, index=customers)

    for cat in GROWTH_CATEGORIES:
        key = _safe_name(cat)
        prev_spend = column(prev, cat)
        typical = column(total, cat) / quarters_seen
        # growth needs +R20 and more than 25% up on last quarter
        needed = np.maximum(prev_spend.clip(lower=0) * 1.25, prev_spend + 20)
        f[f"x_g_{key}_prev"] = prev_spend
        f[f"x_g_{key}_typical_q"] = typical
        f[f"x_g_{key}_prev_vs_typ"] = np.log1p(prev_spend.clip(lower=0)) - np.log1p(typical.clip(lower=0))
        f[f"x_g_{key}_need_vs_typ"] = np.log1p(needed) - np.log1p(typical.clip(lower=0))

    other = [c for c in total.columns if c not in GROWTH_CATEGORIES]
    other_prev = prev[[c for c in prev.columns if c not in GROWTH_CATEGORIES]].sum(axis=1).reindex(customers).fillna(0)
    other_typical = total[other].sum(axis=1).reindex(customers).fillna(0) / quarters_seen
    f["x_g_rest_prev_vs_typ"] = np.log1p(other_prev.clip(lower=0)) - np.log1p(other_typical.clip(lower=0))
    f["x_quarters_seen"] = quarters_seen
    return f
