import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Resolve project root dynamically (folder containing this file)
project_path = os.path.dirname(os.path.abspath(__file__))

# Load the census data
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Split the data into train and test
# (You could use cross-validation instead, but train/test split satisfies the rubric.)
train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

label = "salary"

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label=label,
    training=True,
)

# Process the test data (reuse fitted encoder + label binarizer)
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label=label,
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Ensure model directory exists
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

# Save the model, encoder, and label binarizer
model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)

encoder_path = os.path.join(model_dir, "encoder.pkl")
save_model(encoder, encoder_path)

lb_path = os.path.join(model_dir, "lb.pkl")
save_model(lb, lb_path)

# Reload the model (just to prove load/save works)
model = load_model(model_path)

# Run inference on the test set
preds = inference(model, X_test)

# Calculate and print overall metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Clear previous slice_output file (if any)
slice_output_file = os.path.join(project_path, "slice_output.txt")
open(slice_output_file, "w").close()

# Compute performance on slices of the data
for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label=label,
            encoder=encoder,
            lb=lb,
            model=model,
        )
        with open(slice_output_file, "a", encoding="utf-8") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(
                f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}",
                file=f,
            )
            print("-" * 40, file=f)
