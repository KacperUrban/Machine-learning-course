from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, train_lr, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs="prepared_pokemons",
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data"
        ),
        node(
            func=train_lr,
            inputs=["X_train", "y_train"],
            outputs=["clf_lr", "name_lr"],
            name="train_logistic_regression"
        ),
        node(
            func=evaluate_model,
            inputs=["clf_lr", "X_test", "y_test", "name_lr"],
            outputs=None,
            name="evaluate_model"
        ),
    ])