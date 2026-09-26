from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_and_validate, train_final


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                train_and_validate,
                inputs=["train_table", "label_config", "parameters"],
                outputs=["validation_models", "validation_predictions", "validation_metrics"],
                name="train_and_validate",
            ),
            node(
                train_final,
                inputs=["train_table", "parameters"],
                outputs="final_models",
                name="train_final",
            ),
        ]
    )
