import pandas as pd

from sasol_clv.pipelines.modelling.nodes import predict


def make_submission(final_models: dict, test_table: pd.DataFrame, sample_submission: pd.DataFrame) -> pd.DataFrame:
    preds = predict(final_models, test_table)

    # keep the same IDs and column order as the sample file
    sub = sample_submission[["ID"]].merge(preds, on="ID", how="left")

    # quick checks before we upload anything
    assert len(sub) == len(sample_submission), "row count does not match the sample submission"
    assert sub["ID"].is_unique, "duplicate IDs in submission"
    assert not sub.isna().any().any(), "missing predictions in submission"
    assert list(sub.columns) == list(sample_submission.columns), "columns do not match the sample submission"
    return sub
