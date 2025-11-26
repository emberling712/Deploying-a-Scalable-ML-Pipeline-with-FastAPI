import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Use the same categorical features and label as in train_model.py
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def load_sample_data():
    """
    Helper to load a small sample of the census data for tests.
    Keeps tests fast while still exercising the full pipeline.
    """
    df = pd.read_csv("data/census.csv")
    # Use a subset for speed; 10% of data is usually enough
    sample, _ = train_test_split(df, test_size=0.9, random_state=42)
    return sample


def test_process_data_output_shapes_and_types():
    """
    Test that process_data returns X and y with matching lengths,
    non-zero rows, and non-null encoder and label binarizer.
    """
    df = load_sample_data()

    X, y, encoder, lb = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    # Shapes
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0

    # Artifacts
    assert encoder is not None
    assert lb is not None


def test_train_model_returns_random_forest():
    """
    Test that train_model returns a RandomForestClassifier,
    which is the expected algorithm for this project.
    """
    df = load_sample_data()

    X_train, y_train, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference_and_metrics_behave_sensibly():
    """
    Test that inference returns the correct number of predictions
    and that compute_model_metrics returns values between 0 and 1.
    """
    df = load_sample_data()

    X_train, y_train, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    # Inference length check
    assert len(preds) == len(y_train)

    # Metrics range check
    precision, recall, f1 = compute_model_metrics(y_train, preds)
    for metric in (precision, recall, f1):
        assert 0.0 <= metric <= 1.0
