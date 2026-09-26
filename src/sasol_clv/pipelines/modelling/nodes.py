import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_squared_error

log = logging.getLogger(__name__)

NON_FEATURES = {"ID", "cutoff", "y_fuel", "y_nonfuel", "CLV_fuel", "CLV_nonfuel", "Opportunity"}


def feature_columns(table: pd.DataFrame) -> list:
    return [c for c in table.columns if c not in NON_FEATURES]


def regression_columns(table: pd.DataFrame) -> list:
    # the x_ columns only helped the classifier, the regressors do slightly better without them
    return [c for c in feature_columns(table) if not c.startswith("x_")]


def class_weights(classes, boost: dict) -> np.ndarray:
    big = {"Stable", "Inactivity", "Existing-category growth: Fuel"}
    weights = []
    for c in classes:
        if c in big:
            weights.append(boost["big"])
        elif c.startswith("Existing-category growth"):
            weights.append(boost["other_growth"])
        else:
            weights.append(boost["adoption"])
    return np.array(weights)


def _fit(train: pd.DataFrame, params: dict) -> dict:
    reg_cols = regression_columns(train)
    clf_cols = feature_columns(train)
    seeds = params["seeds"]

    models = {"reg_features": reg_cols, "clf_features": clf_cols, "class_boost": params["class_boost"]}

    # a few seeds per target, their average is a little steadier than one model
    for target, name in [("CLV_fuel", "fuel"), ("CLV_nonfuel", "nonfuel")]:
        models[name] = [
            HistGradientBoostingRegressor(random_state=s, **params["regressor"]).fit(train[reg_cols], train[target])
            for s in seeds
        ]

    # the first cutoff (Jun 2024) only has a partial previous quarter because the data starts on 19 Mar,
    # so its growth labels are too easy to hit. the classifier skips it, the regressors keep it.
    clf_rows = pd.to_datetime(train["cutoff"]) >= pd.Timestamp(params["classifier_min_cutoff"])
    models["opportunity"] = HistGradientBoostingClassifier(random_state=seeds[0], **params["classifier"]).fit(
        train.loc[clf_rows, clf_cols], train.loc[clf_rows, "Opportunity"]
    )
    return models


def predict(models: dict, table: pd.DataFrame) -> pd.DataFrame:
    X_reg = table[models["reg_features"]]
    X_clf = table[models["clf_features"]]
    out = pd.DataFrame({"ID": table["ID"].values})

    # the true values are never below zero after the transform
    for name, col in [("fuel", "CLV_fuel"), ("nonfuel", "CLV_nonfuel")]:
        preds = np.mean([m.predict(X_reg) for m in models[name]], axis=0)
        out[col] = np.clip(preds, 0, None)

    clf = models["opportunity"]
    proba = clf.predict_proba(X_clf) * class_weights(clf.classes_, models["class_boost"])
    out["Opportunity"] = clf.classes_[proba.argmax(axis=1)]
    return out


def score(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    rmse_fuel = float(np.sqrt(mean_squared_error(y_true["CLV_fuel"], y_pred["CLV_fuel"])))
    rmse_nonfuel = float(np.sqrt(mean_squared_error(y_true["CLV_nonfuel"], y_pred["CLV_nonfuel"])))
    f1 = float(f1_score(y_true["Opportunity"], y_pred["Opportunity"], average="weighted"))
    return {
        "rmse_fuel": rmse_fuel,
        "rmse_nonfuel": rmse_nonfuel,
        "weighted_f1": f1,
        # formula as written on the zindi page
        "zindi_score": 0.3 * rmse_fuel + 0.3 * rmse_nonfuel + 0.4 * f1,
        # same idea but every part is an error, so lower is always better
        "error_score": 0.3 * rmse_fuel + 0.3 * rmse_nonfuel + 0.4 * (1 - f1),
    }


def train_and_validate(train_table: pd.DataFrame, label_config: dict, params: dict):
    val_cutoff = pd.Timestamp(params["validation_cutoff"])
    cutoffs = pd.to_datetime(train_table["cutoff"])

    # a training cutoff is safe only if its 3 month future ends before the validation cutoff
    safe = cutoffs + pd.DateOffset(months=3) <= val_cutoff
    train = train_table[safe]
    valid = train_table[cutoffs == val_cutoff]
    log.info("validation: %s train rows, %s valid rows", len(train), len(valid))

    models = _fit(train, params)
    preds = predict(models, valid)
    metrics = {"model": score(valid, preds)}

    # simple baselines so we know the model is worth it
    # baseline 1: next quarter looks exactly like the last one
    s = label_config["normalization"]
    last_q = pd.DataFrame(
        {
            "CLV_fuel": np.log1p(valid["fuel_litres_q1"].clip(lower=0)).values / s["CLV_fuel"],
            "CLV_nonfuel": np.log1p(valid["nonfuel_q1"].clip(lower=0)).values / s["CLV_nonfuel"],
            "Opportunity": "Stable",
        }
    )
    metrics["baseline_last_quarter"] = score(valid, last_q)
    # baseline 2: the sample submission (all zeros and Stable)
    metrics["baseline_zeros"] = score(
        valid, pd.DataFrame({"CLV_fuel": 0.0, "CLV_nonfuel": 0.0, "Opportunity": ["Stable"] * len(valid)})
    )
    metrics["n_train_rows"] = int(len(train))
    metrics["n_valid_rows"] = int(len(valid))

    preds = preds.merge(
        valid[["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]], on="ID", suffixes=("_pred", "_true")
    )
    for name, m in metrics.items():
        if isinstance(m, dict):
            log.info("%s: %s", name, {k: round(v, 4) for k, v in m.items()})
    return models, preds, metrics


def train_final(train_table: pd.DataFrame, params: dict) -> dict:
    # refit on every labelled cutoff, including the validation quarter
    return _fit(train_table, params)
