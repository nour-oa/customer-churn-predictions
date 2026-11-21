# Customer Churn Prediction

Predicts which telecom customers are about to leave, using the open **Telco Customer Churn** dataset (IBM sample data, ~7,000 customers).

## Problem
A company keeps more revenue by retaining customers than by finding new ones. Given a customer's contract, services and billing, the model estimates the probability that they churn, so the company can contact the risky ones first.

## Dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle ("Telco Customer Churn" by BlastChar) and put it in a `data/` folder (not committed to git).

## Method
1. **Cleaning:** `TotalCharges` is stored as text with a few blank rows (new customers); converted to numbers and the blank rows dropped.
2. **Exploration:** churn rate by contract type, internet service, tenure and monthly charges.
3. **Split:** 80/20 stratified train/test split, done before any preprocessing.
4. **Pipeline:** scaling for numeric columns, one-hot encoding for categorical ones, inside a scikit-learn `Pipeline` (no data leakage in cross-validation).
5. **Model comparison:** logistic regression, random forest, gradient boosting, with 5-fold stratified cross-validation on the training set. Compared on precision, recall and F1, not only accuracy, because only about a quarter of customers churn.
6. **Threshold:** chosen on out-of-fold predictions from the training set only.
7. **Final evaluation:** once on the test set, with confusion matrix, precision-recall curve and permutation feature importance.

## Results
> Fill in after running the script on the real dataset.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | | | | |
| Random forest | | | | |
| Gradient boosting | | | | |

Test set (best model): precision __, recall __, F1 __.

Key findings (write 3 sentences in your own words, e.g. which contract type churns most, what tenure shows, which features matter most).

Charts are saved in `outputs/`.

## How to run
```bash
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python churn_prediction.py
```

## Limitations
- Single dataset snapshot, no time dimension.
- The threshold trades missed churners against false alarms; the right value depends on the cost of a retention offer.
- No hyperparameter tuning yet (possible next step: `GridSearchCV`).


Contract type: Customers on month-to-month contracts leave the most (42.7%), compared with 11.3% on one-year contracts and 2.8% on two-year contracts.
Tenure: Churn is highest in the first 12 months (47.4%) and falls steadily with time, to 9.5% for customers with more than 48 months.
Model: The random forest finds contract type, internet service and tenure the most important features, which matches the patterns above.