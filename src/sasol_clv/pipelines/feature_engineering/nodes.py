import pandas as pd

from .features import build_features
from .labels import build_labels, eligible_customers, to_z


def build_train_table(transactions: pd.DataFrame, label_config: dict, params: dict) -> pd.DataFrame:
    s_fuel = label_config["normalization"]["CLV_fuel"]
    s_nonfuel = label_config["normalization"]["CLV_nonfuel"]
    min_tx = label_config["minimum_historical_transactions"]

    frames = []
    for cutoff in params["train_cutoffs"]:
        # only customers that would have been scored at this cutoff
        customers = eligible_customers(transactions, pd.Timestamp(cutoff), min_tx)
        labels = build_labels(transactions, cutoff, label_config, customers)
        feats = build_features(
            transactions, cutoff, customers, params["feature_windows"], params["focus_categories"]
        )
        table = feats.merge(labels, on="ID", how="left")
        table["cutoff"] = cutoff
        frames.append(table)

    train = pd.concat(frames, ignore_index=True)
    train["CLV_fuel"] = to_z(train["y_fuel"], s_fuel)
    train["CLV_nonfuel"] = to_z(train["y_nonfuel"], s_nonfuel)
    return train


def build_test_table(transactions: pd.DataFrame, raw_test: pd.DataFrame, params: dict) -> pd.DataFrame:
    cutoff = params["test_cutoff"]
    feats = build_features(
        transactions, cutoff, raw_test["ID"].unique(), params["feature_windows"], params["focus_categories"]
    )
    feats["cutoff"] = cutoff
    return feats
