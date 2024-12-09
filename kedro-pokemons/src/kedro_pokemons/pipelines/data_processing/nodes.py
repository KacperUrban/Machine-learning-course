import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def remove_incorrect_data(pokemons: pd.DataFrame) -> pd.DataFrame:
    """This function remove any incorrect values and fill missing values with Other class.

    Args:
        pokemons (pd.DataFrame): it contains information about pokemons stats etc. in pandas Dataframe

    Returns:
        pd.DataFrame: cleaned version of pokemons dataset
    """
    pokemons_no_nans = pokemons.fillna("Other")
    pokemons_clean = pokemons_no_nans[
        (pokemons_no_nans.HP > 0)
        & (pokemons_no_nans.Attack > 0)
        & (pokemons_no_nans.Defense > 0)
    ].reset_index(drop=True)
    return pokemons_clean


def encode_normalize_data(pokemons: pd.DataFrame) -> pd.DataFrame:
    """The goal of this function is to encode three types of categorical columns and normalize numerical features.

    Args:
        pokemons (pd.DataFrame): cleaned version of raw pokemons dataset

    Returns:
        pd.DataFrame: prepared dataframe for models training
    """
    label_enc = LabelEncoder()
    pokemons_cat = pd.DataFrame()
    for column in ["Name", "Type 1", "Type 2"]:
        pokemons_cat[column] = label_enc.fit_transform(pokemons[column])

    pokemons_num = pokemons[
        ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    ]
    minmax_scaler = MinMaxScaler()
    columns = pokemons_num.columns
    np_num = minmax_scaler.fit_transform(pokemons_num)
    pokemons_num_norm = pd.DataFrame(np_num, columns=columns)
    pokemons_prepared = pd.concat(
        [pokemons_num_norm, pokemons_cat, pokemons["Generation"]], axis=1
    )
    return pokemons_prepared
