import pandas as pd

# categories that never count as activity (see LABEL_RULES.md rule 4)
EXCLUDED_CATEGORIES = ["Value added services", "Print media", "Carrier Bags"]


def clean_transactions(raw: pd.DataFrame, label_config: dict) -> pd.DataFrame:
    df = raw.copy()

    # timestamps come with milliseconds, some rows may not, so let pandas work it out
    df["DateTimeZA"] = pd.to_datetime(df["DateTimeZA"], format="mixed")

    # blank category should not happen, but keep it as its own value just in case
    df["ItemCategoryLevel2"] = df["ItemCategoryLevel2"].fillna("MISSING")

    # handy flags we use everywhere later
    df["is_adjustment"] = df["ItemName"].isin(label_config["adjustment_items"])
    df["is_excluded_cat"] = df["ItemCategoryLevel2"].isin(EXCLUDED_CATEGORIES)

    # fuel litres for the regression target
    df["is_fuel_litres"] = (df["ItemCategoryLevel2"] == "Fuel") & (df["LineItemType"] == "Sale_Fuel")

    # rows that count towards CLV_nonfuel
    df["is_nonfuel"] = (
        (df["ItemCategoryLevel2"] != "Fuel") & ~df["is_excluded_cat"] & ~df["is_adjustment"]
    )

    # rows that can be an opportunity candidate (fuel included here)
    df["in_scope"] = ~df["is_excluded_cat"] & ~df["is_adjustment"]

    # a qualifying purchase needs positive quantity and positive amount
    df["is_qualifying"] = df["in_scope"] & (df["Quantity"] > 0) & (df["TotalExclAmount"] > 0)

    # we do not drop duplicate lines here, the official labels are built on the raw rows
    df = df.sort_values(["ID", "DateTimeZA", "TransactionId", "LineItemSequence"]).reset_index(drop=True)
    return df
