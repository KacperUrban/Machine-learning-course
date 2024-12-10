from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, train_lr, evaluate_model, train_dt, train_knn, train_rf


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["prepared_pokemons", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=train_lr,
                inputs=["X_train", "y_train"],
                outputs=["clf_lr", "name_lr"],
                name="train_logistic_regression",
            ),
            node(
                func=train_dt,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs=["clf_dt", "name_dt"],
                name="train_decision_tree",
            ),
            node(
                func=train_knn,
                inputs=["X_train", "y_train"],
                outputs=["clf_knn", "name_knn"],
                name="train_knn",
            ),
            node(
                func=train_rf,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs=["clf_rf", "name_rf"],
                name="train_random_forest",
            ),
            node(
                func=evaluate_model,
                inputs=["clf_lr", "X_test", "y_test", "name_lr"],
                outputs=None,
                name="evaluate_model_lr",
            ),
            node(
                func=evaluate_model,
                inputs=["clf_dt", "X_test", "y_test", "name_dt"],
                outputs=None,
                name="evaluate_model_dt",
            ),
            node(
                func=evaluate_model,
                inputs=["clf_knn", "X_test", "y_test", "name_knn"],
                outputs=None,
                name="evaluate_model_knn",
            ),
            node(
                func=evaluate_model,
                inputs=["clf_rf", "X_test", "y_test", "name_rf"],
                outputs=None,
                name="evaluate_model_rf",
            ),
        ]
    )
