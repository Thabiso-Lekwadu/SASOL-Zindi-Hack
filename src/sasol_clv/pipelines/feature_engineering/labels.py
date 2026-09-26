# rebuilds the three targets at any cutoff, following LABEL_RULES.md
# this lets us create labelled training rows from the history in train.csv
import numpy as np
import pandas as pd


def window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df["DateTimeZA"] >= start) & (df["DateTimeZA"] < end)]


def eligible_customers(tx: pd.DataFrame, cutoff, min_tx: int = 3) -> pd.Index:
    # rule 1: at least 3 distinct baskets before the cutoff, any type of row
    hist = tx[tx["DateTimeZA"] < cutoff]
    counts = hist.groupby("ID")["TransactionId"].nunique()
    return counts[counts >= min_tx].index


def build_labels(tx: pd.DataFrame, cutoff, label_config: dict, customers=None) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    fut_end = cutoff + pd.DateOffset(months=3)
    prev_start = cutoff - pd.DateOffset(months=3)

    if customers is None:
        customers = eligible_customers(tx, cutoff, label_config["minimum_historical_transactions"])
    customers = pd.Index(customers, name="ID")

    hist = tx[(tx["DateTimeZA"] < cutoff) & tx["ID"].isin(customers)]
    fut = window(tx, cutoff, fut_end)
    fut = fut[fut["ID"].isin(customers)]
    prev = window(hist, prev_start, cutoff)

    out = pd.DataFrame(index=customers)

    # regression targets, raw units
    fuel = fut[fut["is_fuel_litres"]].groupby("ID")["Quantity"].sum()
    nonfuel = fut[fut["is_nonfuel"]].groupby("ID")["TotalExclAmount"].sum()
    out["y_fuel"] = fuel.reindex(customers).fillna(0.0).values
    out["y_nonfuel"] = nonfuel.reindex(customers).fillna(0.0).values

    # opportunity label
    out["Opportunity"] = build_opportunity(hist, prev, fut, customers, label_config)
    return out.reset_index()


def build_opportunity(hist, prev, fut, customers, cfg) -> pd.Series:
    mapping = cfg["category_mapping"]
    g_min = cfg["growth_min_rand"]
    g_rel = cfg["growth_min_relative"]

    fut_in = fut[fut["in_scope"]]
    prev_in = prev[prev["in_scope"]]

    # what the customer bought before, all history, qualifying only
    seen = hist[hist["is_qualifying"]][["ID", "ItemCategoryLevel2"]].drop_duplicates()
    seen["seen"] = True

    # future spend (net, returns included) and future qualifying baskets per category
    fut_spend = fut_in.groupby(["ID", "ItemCategoryLevel2"])["TotalExclAmount"].sum().rename("fut_spend")
    fut_n = (
        fut_in[fut_in["is_qualifying"]]
        .groupby(["ID", "ItemCategoryLevel2"])["TransactionId"]
        .nunique()
        .rename("fut_n")
    )
    prev_spend = prev_in.groupby(["ID", "ItemCategoryLevel2"])["TotalExclAmount"].sum().rename("prev_spend")

    cand = pd.concat([fut_spend, fut_n], axis=1).reset_index()
    cand = cand.merge(prev_spend.reset_index(), on=["ID", "ItemCategoryLevel2"], how="left")
    cand = cand.merge(seen, on=["ID", "ItemCategoryLevel2"], how="left")
    cand["fut_n"] = cand["fut_n"].fillna(0)
    cand["fut_spend"] = cand["fut_spend"].fillna(0.0)
    cand["prev_spend"] = cand["prev_spend"].fillna(0.0)
    cand["seen"] = cand["seen"].eq(True)

    # growth: bought before, bought again, +R20 and at least 25% up on last quarter
    inc = cand["fut_spend"] - cand["prev_spend"]
    is_growth = (
        cand["seen"]
        & (cand["fut_n"] >= cfg["growth_min_purchases"])
        & (inc >= g_min)
        & (cand["fut_spend"] > (1 + g_rel) * cand["prev_spend"].clip(lower=0))
    )
    # adoption: never bought this category, now buys it with positive net spend
    is_adopt = (
        ~cand["seen"]
        & (cand["fut_n"] >= cfg["adoption_min_purchases"])
        & (cand["fut_spend"] > cfg["adoption_min_rand"])
    )

    growth = cand[is_growth].assign(kind="Existing-category growth", increment=inc[is_growth], kind_rank=1)
    adopt = cand[is_adopt].assign(kind="New category adoption", increment=cand.loc[is_adopt, "fut_spend"], kind_rank=0)
    winners = pd.concat([growth, adopt], ignore_index=True)

    # biggest increment wins, ties go to adoption then alphabetical category
    winners = winners.sort_values(
        ["ID", "increment", "kind_rank", "ItemCategoryLevel2"], ascending=[True, False, True, True]
    ).drop_duplicates("ID")

    def to_label(row):
        cat = row["ItemCategoryLevel2"]
        mapped = mapping.get(cat, {}).get(row["kind"], "Other")
        return f"{row['kind']}: {mapped}"

    labels = pd.Series(index=customers, dtype=object)
    if len(winners):
        labels.loc[winners["ID"].values] = winners.apply(to_label, axis=1).values

    # no candidate: active customers are Stable, everyone else is Inactivity
    active_ids = set(fut_in.loc[fut_in["is_qualifying"], "ID"])
    missing = labels.isna()
    labels[missing] = np.where(labels.index[missing].isin(active_ids), "Stable", "Inactivity")
    return labels.values


def to_z(y, s):
    # official transform, negative totals are clipped to 0 before the log
    return np.log1p(np.clip(y, 0, None)) / s


def from_z(z, s):
    return np.expm1(z * s)
