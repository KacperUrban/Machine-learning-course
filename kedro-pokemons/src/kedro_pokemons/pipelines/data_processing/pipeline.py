from kedro.pipeline import Pipeline, node, pipeline
from .nodes import remove_incorrect_data, encode_normalize_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=remove_incorrect_data,
                inputs="pokemons",
                outputs="cleaned_pokemons",
                name="remove_incorrect_data",
            ),
            node(
                func=encode_normalize_data,
                inputs="cleaned_pokemons",
                outputs="prepared_pokemons",
                name="encode_and_normalize_data",
            ),
        ]
    )
