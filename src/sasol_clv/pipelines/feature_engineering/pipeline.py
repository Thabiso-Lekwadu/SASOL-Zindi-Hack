from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_test_table, build_train_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                build_train_table,
                inputs=["transactions", "label_config", "parameters"],
                outputs="train_table",
                name="build_train_table",
            ),
            node(
                build_test_table,
                inputs=["transactions", "raw_test", "parameters"],
                outputs="test_table",
                name="build_test_table",
            ),
        ]
    )
