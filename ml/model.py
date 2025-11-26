import pickle
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : Any
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model: Any, path: str) -> None:
    """
    Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder, LabelBinarizer, etc.
    path : str
        Path to save pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> Any:
    """
    Loads pickle file from `path` and returns it.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def performance_on_categorical_slice(
    data: pd.DataFrame,
    column_name: str,
    slice_value: Any,
    categorical_features,
    label: str,
    encoder,
    lb,
    model,
):
    """
    Computes the model metrics on a slice of the data specified by a column name
    and slice value.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features.
    label : str
        Name of the label column in `data`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    model : Any
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    # Filter the data to the chosen slice
    slice_df = data[data[column_name] == slice_value]

    # Process the slice (no retraining => training=False)
    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Predict and compute metrics
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
