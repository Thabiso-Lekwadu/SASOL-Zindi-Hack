from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_transactions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                clean_transactions,
                inputs=["raw_transactions", "label_config"],
                outputs="transactions",
                name="clean_transactions",
            ),
        ]
    )
