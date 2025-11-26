# Model Card

This documents the model developed for the Udacity ML Devops projec "Deploying a Machine Learning model with Fast API". This model is made to predict whether or not an individual's annual income exceeds $50k based on provided US Census data. This document describes the model, its use cases, data sources, metrics, ethical considerations and limitations.

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a random forest classifier implemented using the scikit-learn library. It uses the following:
- OneHotEncoder for categorical processing
- **LabelBinarizer** for label encoding  
- A set of categorical and numerical features extracted from the Census Bureau’s adult income dataset  
- A traditional machine learning pipeline consisting of:
  - Data preprocessing (`process_data`)
  - Model training (`train_model`)
  - Inference (`inference`)
  - Model evaluation (`compute_model_metrics`)
  - Slice-based evaluation (`performance_on_categorical_slice`)

  The final model and encoders are stored as:

- `model/model.pkl` — trained RandomForestClassifier  
- `model/encoder.pkl` — trained OneHotEncoder  
- `model/lb.pkl` — trained LabelBinarizer  

## Intended Use

This model is intended to be used for binary income classification using structured demographic and employment data, and it's meant to showcase ML pipelines, showing end to end machine learning deployment with FastAPI.

This is not intended to do any real-world decision making or for use in making actual credit assessments or employment screenings. This is a student project.

## Training Data

The training dataset is the US Census Bureau Adult Income dataset provided inthe Udacity starter repository ('data/census.csv'). It contains features like age, work class, education, marital status, occupation, relationship, race, sex, capital gains and losses, hours worked per week, and native country. 

Preprocessing steps include training and test splitting, one hot encoding of categorical features, converting labels to binary values, and concatenating all transofrmed features into a numerical matrix.

## Evaluation Data

The evaluation dataset consists of the **20% test split** produced from the original Census dataset using:

```python
train_test_split(data, test_size=0.20, random_state=42)

The same preprocessing pipeline applied to the training data is used for the test set, ensuring consistent handling of categorical variables and labels.

In addition to overall evaluation, the model’s performance was measured across categorical data slices, such as education categories, workclass categories, race, sex, and native-country categories. 

These results are saved in slice_output.txt.

## Metrics

The model is evaluated using precision, recall, and F1 score.

These metrics are chosen because the dataset is moderately imbalanced and classification errors have asymmetric implications (e.g., predicting >50K incorrectly).

Overall Model Performance (from train_model.py):

Precision: 0.7483

Recall: 0.6359

F1 Score: 0.6875

Slice-based metrics for each categorical feature and value are included in slice_output.txt. These metrics help identify whether the model’s performance varies significantly across demographic groups — an important step for fairness and transparency.

## Ethical Considerations
This model uses real demographic attributes such as race, sex, marital status, education, and native country.

These attributes correlate with sensitive socioeconomic factors. There is a risk that the model cound reinforce unfair biases present in the training data, or that it could perform unequally across demographic slices.

The slice-based analysis approach helps to expose performance disparities, but it does not completely mitigate them.

## Caveats and Recommendations
- The model uses default Random Forest hyperparameters and could benefit from further tuning.

- No fairness interventions (reweighting, bias mitigation algorithms) were applied.

- The dataset itself has known historical biases and limited feature representation.

- Performance may vary significantly across demographic slices; these should be reviewed before any extended use.

- The model is intended for instructional use only and is not suitable for production without significant additional validation, monitoring, and fairness evaluation.


