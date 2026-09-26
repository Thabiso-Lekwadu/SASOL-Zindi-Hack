from kedro.pipeline import Pipeline, node, pipeline

from .nodes import make_submission


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                make_submission,
                inputs=["final_models", "test_table", "sample_submission"],
                outputs="submission",
                name="make_submission",
            ),
        ]
    )
