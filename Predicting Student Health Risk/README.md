# Predicting Student Health Risk

Playground Series - Season 6 Episode 7

[![Kaggle
Competition](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E7-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e7)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-189FDD)](https://xgboost.readthedocs.io/)

## Predicting Student Health Risk

**Late Submission**

## Overview

**Welcome to the 2026 Kaggle Playground Series!** We plan to continue in
the spirit of previous playgrounds, providing interesting and
approachable datasets for our community to practice their machine
learning skills.

**Your Goal:** Predict student health risk.

This project uses the Kaggle Playground Series Season 6 Episode 7
dataset to build a multiclass machine learning model that predicts one
of three student health conditions:

-   `at-risk`
-   `unhealthy`
-   `fit`

The final model uses **XGBoost** with randomized hyperparameter
optimization using `RandomizedSearchCV`.

## Competition

**Competition:** Playground Series - Season 6 Episode 7

**Start:** July 1, 2026

**Close:** August 1, 2026

### Evaluation

Submissions are evaluated using **balanced accuracy** between the
predicted class and the observed target.

The model's hyperparameter search was therefore optimized using:

``` python
scoring="balanced_accuracy"
```

## Submission File

For each `id` in the test set, the model predicts one of the following
labels for the `health_condition` variable:

``` text
at-risk
unhealthy
fit
```

The submission file contains:

``` text
id,health_condition
690088,at-risk
690089,at-risk
690090,at-risk
etc.
```

The final submission was generated with:

``` python
submission = pd.DataFrame({
    "id": test_df["id"],
    "health_condition": test_predictions
})

submission.to_csv("submission.csv", index=False)
```

## Dataset

The training dataset contains:

``` text
690,088 rows
15 columns
```

The test dataset contains:

``` text
295,753 rows
14 columns
```

The training data includes the target column `health_condition`, while
the test data does not.

## Features

The dataset contains health, lifestyle, activity, and demographic
information.

### Numerical Features

-   `sleep_duration`
-   `heart_rate`
-   `bmi`
-   `calorie_expenditure`
-   `step_count`
-   `exercise_duration`
-   `water_intake`

### Categorical Features

-   `diet_type`
-   `stress_level`
-   `sleep_quality`
-   `physical_activity_level`
-   `smoking_alcohol`
-   `gender`

### Target

The target variable is:

``` text
health_condition
```

It contains three classes:

``` text
at-risk
fit
unhealthy
```

## Data Exploration

The notebook includes exploratory analysis of the training and test
datasets.

The following checks were performed:

-   Dataset shape
-   Data types
-   Missing values
-   Duplicate values
-   Numerical feature statistics
-   Categorical feature values
-   Target classes
-   Feature distributions
-   Feature correlations

There were no duplicate rows in either the training or test dataset.

## Missing Values

The original dataset contains missing values in both numerical and
categorical features.

Examples include:

``` text
sleep_duration
heart_rate
bmi
calorie_expenditure
step_count
exercise_duration
water_intake
diet_type
stress_level
sleep_quality
physical_activity_level
smoking_alcohol
gender
```

Missing values were handled during preprocessing so that the final
training and test matrices contained no null values.

## Data Preprocessing

### 1. Target Encoding

The target labels were converted to numerical values using
`LabelEncoder`.

The mapping used by the notebook was:

``` text
at-risk   → 0
fit       → 1
unhealthy → 2
```

Code:

``` python
le = LabelEncoder()

y_train = le.fit_transform(train["health_condition"])
```

### 2. Remove ID and Target

The `id` column was removed from the model features.

The target column was also removed from the training feature matrix:

``` python
X_train = train.drop(columns=["id", "health_condition"])
X_test = test.drop(columns=["id"])
```

### 3. Ordinal Encoding

The following categorical features have an ordered relationship and were
encoded using `OrdinalEncoder`:

``` text
stress_level
sleep_quality
physical_activity_level
smoking_alcohol
```

The category orders were:

``` text
stress_level:
low → medium → high

sleep_quality:
poor → average → good

physical_activity_level:
sedentary → moderate → active

smoking_alcohol:
no → occasional → yes
```

### 4. One-Hot Encoding

The following nominal categorical features were one-hot encoded:

``` text
diet_type
gender
```

Code:

``` python
X_train = pd.get_dummies(
    X_train,
    columns=["diet_type", "gender"],
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=["diet_type", "gender"],
    dtype=int
)
```

After preprocessing, both training and test feature matrices contained:

``` text
690,088 × 17
295,753 × 17
```

respectively.

## Feature Scaling

Standardization was also explored using `StandardScaler`:

``` python
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
```

The final XGBoost hyperparameter search was performed using the encoded
feature matrices.

## Machine Learning Model

### XGBoost Classifier

The primary model used in this project is:

``` python
XGBClassifier
```

The model was configured for multiclass classification:

``` python
xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss"
)
```

The three classes are:

``` text
0 → at-risk
1 → fit
2 → unhealthy
```

## Hyperparameter Optimization

`RandomizedSearchCV` was used to search for an effective XGBoost
configuration.

The search space included:

``` python
param_dist = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.29),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0, 5)
}
```

The search was configured with:

``` python
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring="balanced_accuracy",
    cv=5,
    random_state=42,
    verbose=2,
    n_jobs=-1
)
```

### Search Configuration

-   **Random combinations:** 20
-   **Cross-validation:** 5-fold
-   **Total model fits:** 100
-   **Evaluation metric:** Balanced Accuracy
-   **Parallel processing:** `n_jobs=-1`
-   **Random state:** 42

The search output confirmed:

``` text
Fitting 5 folds for each of 20 candidates, totalling 100 fits
```

## Best Parameters

The best XGBoost configuration found during randomized search was:

``` text
colsample_bytree  = 0.836965827544817
gamma             = 0.23225206359998862
learning_rate     = 0.1861880070514171
max_depth         = 7
min_child_weight  = 9
n_estimators      = 266
subsample         = 0.6053059844639466
```

## Model Performance

The best cross-validation balanced accuracy obtained during
hyperparameter optimization was:

``` text
0.8587213938707781
```

Approximately:

**Best CV Balanced Accuracy: 0.85872**

### Performance Summary

  Metric                                  Score
  ------------------------------- -------------
  **Best CV Balanced Accuracy**     **0.85872**
  Cross-validation                       5-Fold
  Hyperparameter combinations                20
  Total model fits                          100

> **Note:** The notebook records the cross-validation balanced accuracy
> above. It does not report a separate held-out validation balanced
> accuracy or a Kaggle leaderboard score.

## Final Prediction

After selecting the best XGBoost model, predictions were generated for
the test dataset:

``` python
best_xgb = random_search.best_estimator_

test_predictions = best_xgb.predict(X_test)
```

The numerical predictions were converted back to the original class
labels:

``` python
test_predictions = le.inverse_transform(test_predictions)
```

This produced predictions in the original format:

``` text
at-risk
fit
unhealthy
```

## Kaggle Submission

The final submission contains two columns:

``` text
id
health_condition
```

Example:

``` text
id,health_condition
690088,at-risk
690089,unhealthy
690090,fit
```

The final file is saved as:

``` text
submission.csv
```

## Technologies & Libraries

### Programming Language

-   Python

### Data Analysis

-   Pandas
-   NumPy

### Visualization

-   Matplotlib
-   Seaborn

### Machine Learning

-   Scikit-learn
-   XGBoost
-   SciPy

### Techniques

-   Exploratory Data Analysis
-   Missing-value handling
-   Label encoding
-   Ordinal encoding
-   One-hot encoding
-   Feature scaling
-   Correlation analysis
-   Multiclass classification
-   Randomized hyperparameter search
-   5-fold cross-validation
-   Balanced accuracy evaluation

## Project Workflow

``` text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Exploratory Data Analysis
     │
     ├── Missing Value Analysis
     ├── Duplicate Check
     ├── Statistical Analysis
     └── Feature Correlation
     │
     ▼
Data Preprocessing
     │
     ├── Target Label Encoding
     ├── Ordinal Encoding
     ├── One-Hot Encoding
     └── Feature Scaling
     │
     ▼
XGBoost Classifier
     │
     ▼
RandomizedSearchCV
     │
     ├── 20 Parameter Combinations
     └── 5-Fold CV
     │
     ▼
Best XGBoost Model
     │
     ▼
Test Predictions
     │
     ▼
Inverse Label Encoding
     │
     ▼
submission.csv
```

## Repository Structure

``` text
predicting-student-health-risk/
│
├── predicting_student_health_risk.ipynb
├── README.md
├── submission.csv
│
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## How to Run

### 1. Clone the Repository

``` bash
git clone https://github.com/mahidimithun/predicting-student-health-risk.git
```

### 2. Navigate to the Project

``` bash
cd predicting-student-health-risk
```

### 3. Install Dependencies

``` bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn xgboost jupyter
```

### 4. Launch Jupyter Notebook

``` bash
jupyter notebook
```

Open:

``` text
predicting_student_health_risk.ipynb
```

### 5. Run the Notebook

Make sure the Kaggle dataset files are available under:

``` text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Then run the notebook cells sequentially.

## Key Learning Outcomes

This project provided practical experience with:

-   Large-scale tabular datasets
-   Multiclass classification
-   Missing-value analysis
-   Categorical feature encoding
-   Ordinal encoding
-   One-hot encoding
-   Feature scaling
-   XGBoost
-   Randomized hyperparameter optimization
-   Cross-validation
-   Balanced accuracy
-   Label encoding and inverse transformation
-   Kaggle submission preparation

## Future Improvements

Potential improvements for this project include:

### 1. Advanced Feature Engineering

Create additional health-related interaction features such as:

-   Exercise-to-calorie relationships
-   Sleep and physical activity interactions
-   BMI-related groups
-   Hydration and activity ratios
-   Heart-rate and exercise interactions

### 2. Model Comparison

Compare XGBoost with:

-   LightGBM
-   CatBoost
-   Random Forest
-   Extra Trees
-   HistGradientBoosting
-   Logistic Regression

### 3. Ensemble Learning

Combine predictions from multiple strong models using:

-   Soft voting
-   Stacking
-   Blending

### 4. Hyperparameter Optimization

Use more iterations in `RandomizedSearchCV` or explore more advanced
optimization approaches such as Bayesian optimization.

### 5. Model Explainability

Use SHAP or feature importance analysis to understand which health and
lifestyle features contribute most to the predicted health condition.

## Timeline

-   **Start Date:** July 1, 2026
-   **Entry Deadline:** Same as the Final Submission Deadline
-   **Team Merger Deadline:** Same as the Final Submission Deadline
-   **Final Submission Deadline:** July 31, 2026
-   **Competition Close:** August 1, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless
otherwise noted.

## About the Tabular Playground Series

The Tabular Playground Series provides the Kaggle community with
relatively lightweight machine learning challenges for practicing and
improving data science and machine learning skills.

These competitions provide opportunities to:

-   Explore tabular datasets
-   Practice feature engineering
-   Build machine learning models
-   Experiment with different algorithms
-   Create visualizations
-   Improve model performance
-   Participate in Kaggle competitions

## Synthetically-Generated Datasets

The Playground Series uses synthetically generated datasets based on
real-world data. This provides a balance between realistic feature
structures and private test labels, allowing participants to practice
machine learning without access to the hidden test targets.

## Prizes

-   **1st Place** - Choice of Kaggle merchandise
-   **2nd Place** - Choice of Kaggle merchandise
-   **3rd Place** - Choice of Kaggle merchandise

Kaggle notes that merchandise will only be awarded once per person in
this series. If a previous winner is selected, the prize may move to the
next eligible team.

## Citation

Yao Yan, Walter Reade, Elizabeth Park. Predicting Student Health Risk.

https://kaggle.com/competitions/playground-series-s6e7

2026. Kaggle.

## Author

### Mahidi Hasan Mithun

**Computer Science Graduate \| Machine Learning \| Data Science \|
Python**

GitHub:

https://github.com/mahidimithun

## Acknowledgment

This project was developed as part of the **Kaggle Playground Series -
Season 6 Episode 7: Predicting Student Health Risk** competition.

The project focuses on building a complete multiclass machine learning
workflow, including data exploration, preprocessing, feature encoding,
XGBoost modeling, hyperparameter optimization, balanced accuracy
evaluation, and Kaggle submission generation.

## 🔗 Links

-   [Kaggle
    Competition](https://www.kaggle.com/competitions/playground-series-s6e7)
-   [GitHub Profile](https://github.com/mahidimithun)

------------------------------------------------------------------------

⭐ If you find this project useful, consider giving the repository a
star!
