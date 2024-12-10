from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np


def split_data(
    input_data: pd.DataFrame, parameters: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """This function split data into test and train sets

    Args:
        input_data (pd.DataFrame): cleaned pokemons data

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: four sets with train and test data
    """
    X = input_data[parameters["features"]]
    y = input_data[parameters["label"]]
    if parameters["stratify"]:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            stratify=y,
            test_size=parameters["test_size"],
            random_state=parameters["random_state"],
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    return X_train, X_test, y_train, y_test


def train_lr(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LogisticRegression, str]:
    """Function trains a logisitic regression model

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training labels

    Returns:
        tuple[LogisticRegression, str]: trained classifier
    """
    clf_lr = LogisticRegression()
    clf_lr.fit(X_train, y_train)
    return clf_lr, "logistic regression"


def train_rf(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> tuple[RandomForestClassifier, str]:
    """Function trains a random forest model

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training labels

    Returns:
        tuple[RandomForestClassifier, str]: trained classifier
    """
    clf_rf = RandomForestClassifier(random_state=parameters["random_state"])
    clf_rf.fit(X_train, y_train)
    return clf_rf, "random forest"


def train_knn(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[KNeighborsClassifier, str]:
    """Function trains a knn model

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training labels

    Returns:
        tuple[KNeighborsClassifier, str]: trained classifier
    """
    clf_knn = KNeighborsClassifier()
    clf_knn.fit(X_train, y_train)
    return clf_knn, "knn"


def train_dt(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> tuple[DecisionTreeClassifier, str]:
    """Function trains a decision tree model

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training labels

    Returns:
        tuple[DecisionTreeClassifier, str]: trained classifier
    """
    clf_rf = RandomForestClassifier(random_state=parameters["random_state"])
    clf_rf.fit(X_train, y_train)
    return clf_rf, "decision tree"


def evaluate_model(
    clf: (
        LogisticRegression
        | RandomForestClassifier
        | DecisionTreeClassifier
        | KNeighborsClassifier
        | SVC
    ),
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> None:
    """Function evaluate models

    Args:
        clf (LogisticRegression | RandomForestClassifier | DecisionTreeClassifier | KNeighborsClassifier | SVC): trained model
        X_test (pd.DataFrame): test features
        y_test (pd.Series): test labels
        name (str): name of the model
    """
    y_pred = clf.predict(X_test)
    print(f"Accuracy of {name}: {np.round(accuracy_score(y_test, y_pred), 3)}")
    print(
        f"F1 score of {name}: {np.round(f1_score(y_test, y_pred, average="macro"), 3)}"
    )
