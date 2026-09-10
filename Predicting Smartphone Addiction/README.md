# 📱 Predicting Smartphone Addiction

A machine learning project for the **Kaggle Playground Series -- Season
6 Episode 8: Predicting Smartphone Addiction** competition.

The objective of this project is to build a binary classification model
that predicts whether an individual is likely to be classified as
smartphone-addicted based on demographic, smartphone-usage, lifestyle,
stress, and academic/work-related features.

------------------------------------------------------------------------

## 🏆 Competition

**Kaggle:** Playground Series -- Season 6 Episode 8: Predicting
Smartphone Addiction

-   **Task:** Binary Classification
-   **Target:** `addicted_label`
-   **Evaluation Metric:** ROC-AUC
-   **Training samples:** 691,369
-   **Test samples:** 296,302

The training dataset contains **14 columns**, including the target
variable, while the test dataset contains **13 feature columns**.

------------------------------------------------------------------------

## 🎯 Project Objective

The main goal is to predict the probability that an individual belongs
to the positive class:

-   `0` → Not addicted
-   `1` → Addicted

Because the Kaggle competition evaluates predictions using **ROC-AUC**,
the model is optimized for ROC-AUC rather than accuracy alone.

------------------------------------------------------------------------

## 📊 Dataset Features

The dataset contains the following original features:

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `age`                               Age of the individual

  `daily_screen_time_hours`           Daily smartphone screen time

  `social_media_hours`                Daily time spent on social media

  `gaming_hours`                      Daily time spent gaming

  `work_study_hours`                  Daily work/study hours

  `sleep_hours`                       Daily sleep duration

  `notifications_per_day`             Number of notifications received
                                      per day

  `app_opens_per_day`                 Number of smartphone app openings
                                      per day

  `weekend_screen_time`               Smartphone screen time during
                                      weekends

  `gender`                            Gender category

  `stress_level`                      Reported stress level

  `academic_work_impact`              Whether smartphone usage impacts
                                      academic/work activities

  `addicted_label`                    Target variable; available only in
                                      training data
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 📁 Dataset Size

The notebook loads:

``` python
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")
```

Dataset dimensions:

``` text
Train: (691369, 14)
Test:  (296302, 13)
```

There were no duplicate rows in either the training or test dataset.

------------------------------------------------------------------------

## 🔎 Exploratory Data Analysis

The initial analysis included:

-   Inspecting dataset dimensions
-   Checking data types
-   Inspecting missing values
-   Checking duplicate rows
-   Identifying numerical and categorical features
-   Examining the target variable
-   Inspecting categorical feature values

### Numerical Features

``` text
age
daily_screen_time_hours
social_media_hours
gaming_hours
work_study_hours
sleep_hours
notifications_per_day
app_opens_per_day
weekend_screen_time
```

### Categorical Features

``` text
gender
stress_level
academic_work_impact
```

------------------------------------------------------------------------

## 🧹 Data Preprocessing

### 1. Remove ID and Separate Target

The `id` column was removed from the modeling features.

``` python
X = df_train.drop(columns=['id', 'addicted_label'])
y = df_train['addicted_label']
```

The dataset was then divided using a stratified 80/20 split:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

------------------------------------------------------------------------

### 2. Missing-Value Indicators

The dataset contains a substantial number of missing values.

Instead of only replacing missing values, binary missingness indicators
were created for every numerical and categorical feature.

For example:

``` python
X_train[f'{col}_missing'] = X_train[col].isna().astype(int)
```

This allows the model to retain information about whether a particular
value was originally missing.

Missingness indicators were created for:

``` text
age_missing
daily_screen_time_hours_missing
social_media_hours_missing
gaming_hours_missing
work_study_hours_missing
sleep_hours_missing
notifications_per_day_missing
app_opens_per_day_missing
weekend_screen_time_missing
gender_missing
stress_level_missing
academic_work_impact_missing
```

------------------------------------------------------------------------

### 3. Numerical Imputation

Missing numerical values were replaced using the **median** calculated
from the training data.

``` python
num_imputer = SimpleImputer(strategy='median')

X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])
df_test[num_cols] = num_imputer.transform(df_test[num_cols])
```

The imputer was fitted only on the training data and then applied to the
validation/test data.

------------------------------------------------------------------------

### 4. Categorical Imputation

Missing categorical values were replaced using the **most frequent
category**.

``` python
cat_imputer = SimpleImputer(strategy='most_frequent')

X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
df_test[cat_cols] = cat_imputer.transform(df_test[cat_cols])
```

After imputation, the notebook verified that there were no remaining
missing values.

------------------------------------------------------------------------

## 🔤 Categorical Encoding

Different encoding strategies were used depending on the nature of each
categorical feature.

### Stress Level

`stress_level` has an inherent order:

``` text
Low < Medium < High
```

Therefore, ordinal encoding was used:

``` python
stress_order = ['Low', 'Medium', 'High']

ordinal_enc = OrdinalEncoder(
    categories=[stress_order]
)
```

This produces:

``` text
Low    → 0
Medium → 1
High   → 2
```

### Academic/Work Impact

`academic_work_impact` is binary:

``` text
Yes → 1
No  → 0
```

### Gender

`gender` is a nominal categorical feature, so one-hot encoding was used:

``` python
X_train = pd.get_dummies(
    X_train,
    columns=['gender'],
    drop_first=True
)
```

The resulting gender features include:

``` text
gender_Male
gender_Other
```

The test datasets were aligned with the training columns to ensure
consistent feature structure.

------------------------------------------------------------------------

## 🧮 Final Feature Set

After preprocessing, missingness indicators, encoding, and alignment,
the training data contained **25 model features**.

The final feature representation included:

-   Original numerical features
-   Encoded categorical features
-   Missingness indicator features
-   One-hot encoded gender features

Boolean features generated by one-hot encoding were converted to integer
values before model training.

------------------------------------------------------------------------

# 🤖 Machine Learning Model

## Random Forest Classifier

The main machine learning algorithm used in this project is:

``` python
RandomForestClassifier
```

Random Forest was selected because it can effectively model nonlinear
relationships and interactions between features without requiring
feature scaling.

------------------------------------------------------------------------

## 🔧 Hyperparameter Optimization

`RandomizedSearchCV` was used to search for a strong combination of
Random Forest hyperparameters.

The search space included:

``` python
param_dist = {
    'n_estimators': randint(200, 600),
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 0.5],
    'class_weight': ['balanced', None]
}
```

The search configuration was:

``` python
RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=3,
    random_state=42,
    n_jobs=-1
)
```

### Search Details

-   **Random combinations:** 20
-   **Cross-validation:** 3-fold
-   **Total model fits:** 60
-   **Optimization metric:** ROC-AUC
-   **Parallel processing:** `n_jobs=-1`

------------------------------------------------------------------------

## 🏆 Best Hyperparameters

The best Random Forest configuration found during hyperparameter tuning
was:

``` text
n_estimators      = 463
max_depth         = 40
min_samples_split = 13
min_samples_leaf  = 7
max_features      = 0.5
class_weight      = None
```

### Best Cross-Validation Score

``` text
Best CV ROC-AUC: 0.9422493525261051
```

Approximately:

**CV ROC-AUC = 0.94225**

------------------------------------------------------------------------

# 📈 Model Performance

The optimized Random Forest model was evaluated on the held-out 20% test
split.

## Performance Summary

  Metric                 Score
  -------------- -------------
  **ROC-AUC**      **0.94222**
  **Accuracy**      **86.94%**

### ROC-AUC

``` text
0.9422155613809657
```

### Accuracy

``` text
0.8694331544614317
```

Therefore:

> **Local Test Accuracy: 86.94%**

> **Local Test ROC-AUC: 0.94222**

**Important:** 86.94% is the accuracy from the local held-out test split
in this notebook. The official Kaggle competition metric is **ROC-AUC**.

------------------------------------------------------------------------

## 📋 Classification Report

``` text
              precision    recall  f1-score   support

           0       0.80      0.74      0.77     40179
           1       0.90      0.92      0.91     98095

    accuracy                           0.87    138274
   macro avg       0.85      0.83      0.84    138274
weighted avg       0.87      0.87      0.87    138274
```

### Class 0

-   Precision: **0.80**
-   Recall: **0.74**
-   F1-score: **0.77**

### Class 1

-   Precision: **0.90**
-   Recall: **0.92**
-   F1-score: **0.91**

------------------------------------------------------------------------

## 🔢 Confusion Matrix

The resulting confusion matrix was:

``` text
[[29737 10442]
 [ 7612 90483]]
```

Interpreted as:

``` text
                 Predicted
                 0       1

Actual 0       29737   10442
Actual 1        7612   90483
```

The model correctly classified:

-   **29,737** samples from class `0`
-   **90,483** samples from class `1`

------------------------------------------------------------------------

# 🎯 Probability Prediction

Because the Kaggle competition uses ROC-AUC, probability predictions are
generated instead of only class labels.

The notebook uses:

``` python
y_proba = best_rf.predict_proba(X_test)[:, 1]
```

The second probability column represents the predicted probability of
class `1` (`addicted_label = 1`).

------------------------------------------------------------------------

# 📤 Kaggle Submission

The final submission follows the required structure:

``` text
id,addicted_label
691369,0.2
691370,0.3
691371,0.1
```

The `addicted_label` column contains probability values rather than hard
`0`/`1` predictions.

------------------------------------------------------------------------

# 🧰 Technologies Used

### Programming Language

-   Python

### Data Analysis

-   Pandas
-   NumPy

### Machine Learning

-   Scikit-learn
-   Random Forest
-   RandomizedSearchCV
-   SciPy

### Machine Learning Techniques

-   Exploratory Data Analysis
-   Missing-value handling
-   Missingness indicators
-   Median imputation
-   Most-frequent imputation
-   Ordinal encoding
-   One-hot encoding
-   Stratified train-test split
-   Cross-validation
-   Hyperparameter optimization
-   Binary classification
-   Probability prediction
-   ROC-AUC evaluation

------------------------------------------------------------------------

# 📚 Key Learning Outcomes

This project provided practical experience with:

-   Working with large tabular datasets
-   Handling missing values in machine learning
-   Preserving missingness information through indicator variables
-   Choosing appropriate encoding techniques for categorical features
-   Building a Random Forest classifier
-   Performing randomized hyperparameter search
-   Using cross-validation on a large dataset
-   Optimizing a model for ROC-AUC
-   Evaluating binary classification models
-   Generating probability-based predictions
-   Preparing Kaggle submissions

------------------------------------------------------------------------

# 🚀 Future Improvements

Several approaches could potentially improve the model further.

## 1. Gradient Boosting Models

Experiment with:

-   XGBoost
-   LightGBM
-   CatBoost

These models can perform very well on structured/tabular datasets.

## 2. Feature Engineering

Potential additional features could include:

-   Screen-time-to-sleep ratios
-   Social-media-to-total-screen-time ratios
-   Gaming-to-total-screen-time ratios
-   Combined app engagement indicators
-   Weekend vs. weekday usage relationships
-   Interaction features between stress and smartphone usage

## 3. Ensemble Learning

Combine predictions from multiple strong models using:

-   Voting
-   Stacking
-   Blending

## 4. Model Explainability

Use tools such as SHAP to investigate:

-   Which features contribute most to predictions
-   How smartphone usage affects predicted addiction probability
-   How stress and lifestyle features influence the model

## 5. More Extensive Hyperparameter Search

A larger search space and additional iterations could potentially find
an even better model.

------------------------------------------------------------------------

# 📁 Repository Structure

``` text
predicting-smartphone-addiction/
│
├── predicting-smartphone-addiction.ipynb
├── README.md
├── train.csv
├── test.csv
├── sample_submission.csv
└── submission.csv
```

> Large Kaggle datasets may be excluded from the Git repository
> depending on GitHub file-size limitations.

------------------------------------------------------------------------

# ▶️ How to Run

## 1. Clone the Repository

``` bash
git clone https://github.com/mahidimithun/predicting-smartphone-addiction.git
```

## 2. Navigate to the Project

``` bash
cd predicting-smartphone-addiction
```

## 3. Install Dependencies

``` bash
pip install pandas numpy scikit-learn scipy jupyter
```

## 4. Launch Jupyter Notebook

``` bash
jupyter notebook
```

Open:

``` text
predicting-smartphone-addiction.ipynb
```

## 5. Run the Notebook

Make sure the required Kaggle dataset files are available in the
appropriate directory:

``` text
train.csv
test.csv
sample_submission.csv
```

Then run the notebook cells sequentially.

------------------------------------------------------------------------

# 🏅 Results at a Glance

``` text
Model                  Random Forest Classifier
Training Samples       691,369
Test Samples           296,302
CV                     3-Fold
Random Search          20 combinations
Total Fits             60
Best CV ROC-AUC        0.94225
Local Test ROC-AUC     0.94222
Local Test Accuracy    86.94%
```

------------------------------------------------------------------------

# 👨‍💻 Author

## Mahidi Hasan Mithun

**Computer Science Graduate \| Machine Learning \| Data Science \|
Python**

GitHub:\
https://github.com/mahidimithun

------------------------------------------------------------------------

# ⭐ Acknowledgment

This project was developed as part of the **Kaggle Playground Series --
Season 6 Episode 8: Predicting Smartphone Addiction** competition.

The project focuses on developing a complete tabular machine learning
pipeline, from data preprocessing and feature engineering to model
tuning, evaluation, and probability-based prediction.

------------------------------------------------------------------------

## 🔗 Links

-   [Kaggle
    Competition](https://www.kaggle.com/competitions/playground-series-s6e8)
-   [GitHub Profile](https://github.com/mahidimithun)

------------------------------------------------------------------------

## ⭐ If you find this project useful

If you find this project helpful or interesting, consider giving the
repository a ⭐ on GitHub.
