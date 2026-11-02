# Will this visitor buy? - Online shopper purchase prediction

Machine Learning (AIML ZG565) - Assignment 2
Saheb Chatterjee, 2026AJ05643

## a. Problem statement

An online store sees thousands of browsing sessions a day and only a small share of
them end in a purchase. Given what a visitor did during a session (which kinds of pages
they opened, how long they stayed, how they arrived, the month, and so on), predict
whether the session will end in a purchase (`Revenue = True`) or not.

This is a binary classification problem with an imbalanced target: about 15.6 % of
sessions are purchases. Five classifiers are trained on the same data and compared on
Accuracy, AUC, Precision, Recall, F1 and MCC. The trained models are served in a
Streamlit app that evaluates them on an uploaded test file.

## b. Dataset description

**Source:** Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository
(dataset id 468), Sakar et al., 2018. https://archive.ics.uci.edu/dataset/468

- 12,330 sessions, 17 input features, 1 target (`Revenue`).
- No missing values. 125 rows are exact duplicates and were removed before splitting,
  leaving **12,205** sessions: 10,297 without a purchase and 1,908 with one (15.6 %).
- Split: 80 % train (9,764 rows) and 20 % test (2,441 rows), stratified on the target,
  `random_state = 5643`. The test part is `test_data.csv`.

| Feature | Type | Meaning |
|---|---|---|
| Administrative, Informational, ProductRelated | count | Number of pages of each kind visited in the session |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | numeric | Time spent on each kind of page (seconds) |
| BounceRates | numeric | Average bounce rate of the pages visited (Google Analytics) |
| ExitRates | numeric | Average exit rate of the pages visited |
| PageValues | numeric | Average value of the pages visited before a transaction |
| SpecialDay | numeric (0-1) | How close the visit is to a special day such as Valentine's Day |
| Month | categorical | Month of the visit (10 months appear; no January or April) |
| OperatingSystems, Browser, Region, TrafficType | categorical (integer codes) | 8, 13, 9 and 20 codes |
| VisitorType | categorical | Returning_Visitor, New_Visitor or Other |
| Weekend | boolean | Whether the visit was on a weekend |
| **Revenue** | **target** | **Whether the session ended in a purchase** |

**Preprocessing** (fitted on the training part only, saved inside every model pipeline and
also as `model/preprocessor.joblib`):

- the six page counts/durations and `PageValues` are heavily right-skewed (skewness between 1.9 and 7.5),
  so they get `log1p` and then standard scaling;
- `BounceRates`, `ExitRates`, `SpecialDay` are standardised;
- the seven categorical columns are one-hot encoded; codes seen fewer than 20 times are
  grouped into one "infrequent" column; a code never seen in training goes to that
  column too (or is ignored if the column has no infrequent group).

## c. GitHub repository link

**https://github.com/2026aj05643/shopper-intent-classifier**

```
shopper-intent-classifier/
|-- app.py                     Streamlit app
|-- model_training.ipynb       data preparation, training, evaluation, saving
|-- requirements.txt
|-- README.md
|-- test_data.csv              20 % hold-out split (17 features + Revenue)
|-- data/
|   `-- online_shoppers_intention.csv     original UCI file
`-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- svm.joblib
    |-- preprocessor.joblib     fitted preprocessing on its own
    |-- model_info.json         feature list, target, best settings, library versions
    `-- test_metrics.csv        the comparison table below
```

## d. Models used

All five models use the same preprocessing and the same train/test split. A few settings
of each were chosen by 5-fold stratified cross-validation on the training part, scored by
F1 of the purchase class:

| Model | Settings chosen |
|---|---|
| Logistic Regression | C = 0.01, class_weight = balanced |
| Decision Tree | criterion = entropy, max_depth = 6, min_samples_leaf = 5, class_weight = balanced |
| KNN | k = 21, uniform weights (Euclidean distance on the scaled features) |
| Naive Bayes | Gaussian, var_smoothing = 0.1 |
| Support Vector Machine | RBF kernel, C = 1, class_weight = balanced, probability outputs on |

A session is predicted as a purchase when the model's predicted probability is at least
0.5. AUC uses the probabilities directly.

### Comparison table (test set, 2,441 sessions)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8734 | 0.9186 | 0.5667 | 0.8115 | **0.6674** | **0.6068** |
| Decision Tree | 0.8542 | **0.9240** | 0.5211 | 0.8403 | 0.6433 | 0.5830 |
| KNN | **0.9009** | 0.9134 | **0.7465** | 0.5550 | 0.6366 | 0.5892 |
| Naive Bayes | 0.7530 | 0.8700 | 0.3757 | **0.8743** | 0.5256 | 0.4566 |
| Support Vector Machine | 0.8898 | 0.9191 | 0.6452 | 0.6571 | 0.6511 | 0.5857 |

(Bold = best in the column.)

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best F1 (0.667) and best MCC (0.607). With balanced class weights it finds 81 % of the buyers, at the cost of precision 0.57 (roughly 4 in 10 sessions it flags do not buy). Strong regularisation (C = 0.01) worked best, which suggests a few features carry most of the signal: the largest coefficient by far is on `PageValues`, followed by `Month = Nov` (positive) and `Month = May` (negative). Simple, fast and easy to explain. |
| Decision Tree | Highest AUC (0.924), so it ranks sessions slightly better than the others, but at the 0.5 cut-off it flags too many non-buyers (precision 0.52), which pulls its F1 and MCC below Logistic Regression. The tree splits first on `PageValues`, and that one feature carries about 68 % of its importance. Depth 6 with at least 5 sessions per leaf kept it from overfitting; deeper trees scored worse in cross-validation. |
| KNN | Highest accuracy (0.901) and precision (0.747), but it misses about 45 % of the buyers (recall 0.555). With no class weighting, the 21 nearest neighbours of most sessions are mainly non-buyers, so the vote leans towards "no purchase". Its high accuracy mostly reflects the 84 % majority class, which is why accuracy alone is misleading here. It also keeps the whole training set in memory (the largest saved model, about 5 MB). |
| Naive Bayes | Weakest overall (F1 0.526, MCC 0.457, accuracy 0.753). It catches the most buyers (recall 0.874) but more than 6 in 10 of its purchase predictions are wrong (precision 0.376). Its assumptions do not fit this data: features such as page counts and their durations are strongly related, and the one-hot and skewed columns are not Gaussian. The best smoothing was the largest value tried, another sign the Gaussian fit is poor. |
| Support Vector Machine | Close to the top on AUC (0.919) and second on F1 (0.651), with precision and recall almost equal (0.645 and 0.657), so it makes about as many false alarms as misses. It was the slowest to train and to predict (it keeps 3,514 support vectors), with no gain over Logistic Regression on this data. |
| **Overall, winner for your dataset?** | **Logistic Regression.** It has the best F1 and MCC, the two measures that account for both classes on this imbalanced data, and its AUC (0.919) is within 0.006 of the best. The Decision Tree is the runner-up and ranks marginally better (AUC 0.924). If the store cared more about not wasting offers than about catching every buyer, KNN's precision, or Logistic Regression with a higher threshold (try the slider in the app), would be the choice. |

## Streamlit app

Link: **https://shopper-intent-2026aj05643.streamlit.app/**

- **Test data upload (CSV):** upload a file with the 17 feature columns and `Revenue`.
  The `test_data.csv` from this repository can be downloaded from the sidebar and
  uploaded back, or used directly with the checkbox.
- **Model selection:** a dropdown with the five saved classifiers.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model.
- **Confusion matrix and classification report** for the chosen model, plus its ROC curve.
- **All five models** tab: the same metrics for every model on the uploaded file, with
  the best value in each column highlighted, a bar chart and overlaid ROC curves.
- A **decision threshold** slider (default 0.5, the value used in the table above).
- Predictions can be downloaded as CSV.

The app only loads the saved models from `model/` and predicts; it never retrains on the
uploaded data.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# retrain and regenerate model/ and test_data.csv (optional, needs jupyter)
pip install jupyter
jupyter nbconvert --to notebook --execute --inplace model_training.ipynb

# start the app
streamlit run app.py
```

Tested with Python 3.12. The saved models were created with scikit-learn 1.7.2, so
`requirements.txt` pins that version.
