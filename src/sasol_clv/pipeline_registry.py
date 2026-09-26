from kedro.pipeline import Pipeline

from sasol_clv.pipelines import data_processing, feature_engineering, modelling, submission


def register_pipelines() -> dict[str, Pipeline]:
    dp = data_processing.create_pipeline()
    fe = feature_engineering.create_pipeline()
    md = modelling.create_pipeline()
    sb = submission.create_pipeline()

    # kedro run                      -> everything end to end
    # kedro run --pipeline features  -> stop after building features
    return {
        "__default__": dp + fe + md + sb,
        "data_processing": dp,
        "features": dp + fe,
        "modelling": md,
        "submission": sb,
    }
