"""Train and evaluate two reproducible credit-risk baselines."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_credit_data


RANDOM_STATE = 42


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.columns.difference(numeric).tolist()
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def evaluate(y_true, probability, threshold=0.5):
    predictions = (probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(y_true, predictions, output_dict=True, zero_division=0),
    }


def main():
    X, y = load_credit_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    metrics = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        metrics[name] = evaluate(y_test, probability)
        fitted[name] = pipeline
        print(f"{name}: ROC-AUC={metrics[name]['roc_auc']:.3f}; AP={metrics[name]['average_precision']:.3f}")

    # Select model on the test split for demonstration only; production selection
    # must use a separate validation split or cross-validation.
    best_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    (output / "metrics.json").write_text(json.dumps({"results": metrics, "selected_for_demo": best_name}, indent=2), encoding="utf-8")
    joblib.dump(fitted[best_name], output / "model.joblib")
    ConfusionMatrixDisplay(np.array(metrics[best_name]["confusion_matrix"]), display_labels=["good", "bad"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved artifacts in {output.resolve()}")


if __name__ == "__main__":
    main()
