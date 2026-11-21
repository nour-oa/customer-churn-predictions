"""Customer churn prediction on the Telco Customer Churn dataset.

Run:  python churn_prediction.py
Needs: data/WA_Fn-UseC_-Telco-Customer-Churn.csv  (see README for the download link)
Writes charts and result tables to outputs/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT_DIR = Path("outputs")
RANDOM_STATE = 42


# ---------------------------------------------------------------- 1. load + clean
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # TotalCharges is text because a few rows are blank (new customers):
    # convert to numbers, blanks become NaN, then drop those few rows.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)  # 1 = customer left
    return df


# ---------------------------------------------------------------- 2. exploration
def explore(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)} | churn rate: {df['Churn'].mean():.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes[:2], ["Contract", "InternetService"]):
        df.groupby(col)["Churn"].mean().sort_values().plot.bar(ax=ax)
        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("churn rate")
        ax.tick_params(axis="x", rotation=0)
    df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[2])
    axes[2].set_title("Monthly charges (0 = stayed, 1 = left)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_exploration.png", dpi=150)
    plt.close()

    # tenure: customers leave early
    df.assign(tenure_group=pd.cut(df["tenure"], [0, 12, 24, 48, 72], include_lowest=True)).groupby(
        "tenure_group", observed=True
    )["Churn"].mean().plot.bar(title="Churn rate by tenure (months)")
    plt.ylabel("churn rate")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_tenure.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------- 3-5. split, pipeline, models
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def get_models(X: pd.DataFrame) -> dict:
    models = {
        "Logistic regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    # the preprocessing lives INSIDE the pipeline, so cross-validation never
    # leaks information from the validation fold into training
    return {
        name: Pipeline([("prep", build_preprocessor(X)), ("model", m)])
        for name, m in models.items()
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    explore(df)

    X, y = df.drop(columns="Churn"), df["Churn"]
    # stratify keeps the same churn rate in train and test; split BEFORE anything else
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ------------------------------------------------------------ compare models (CV)
    models = get_models(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1"]
    rows = []
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({"model": name, **{m: res[f"test_{m}"].mean() for m in scoring}})
    results = pd.DataFrame(rows).set_index("model").round(3)
    print("\n5-fold cross-validation on the training set:")
    print(results)
    results.to_csv(OUT_DIR / "cv_results.csv")

    best_name = results["f1"].idxmax()
    best = models[best_name]
    print(f"\nBest model by F1: {best_name}")

    # ------------------------------------------------------------ choose threshold (train only)
    # Out-of-fold probabilities on the TRAIN set: pick the threshold with best F1
    # without ever looking at the test set.
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    threshold = float(thr[np.argmax(f1s)])
    print(f"Chosen decision threshold: {threshold:.2f}")

    # ------------------------------------------------------------ final test (once)
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    print("\nTest set results:")
    print(classification_report(y_test, pred, target_names=["stayed", "left"]))
    print(f"Test F1 (churn class): {f1_score(y_test, pred):.3f}")

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["stayed", "left"], cmap="Blues"
    )
    plt.title(f"{best_name} - confusion matrix (test)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_confusion_matrix.png", dpi=150)
    plt.close()

    p_t, r_t, _ = precision_recall_curve(y_test, proba)
    plt.plot(r_t, p_t)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall curve (test)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_precision_recall.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------ feature importance
    imp = permutation_importance(
        best, X_test, y_test, scoring="f1", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    top = (
        pd.Series(imp.importances_mean, index=X_test.columns)
        .sort_values()
        .tail(10)
    )
    top.plot.barh(title="Permutation importance (drop in F1 when shuffled)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_feature_importance.png", dpi=150)
    plt.close()
    print("\nTop features:\n", top.sort_values(ascending=False).round(4))
    print(f"\nCharts and tables saved in {OUT_DIR}/")


if __name__ == "__main__":
    main()
