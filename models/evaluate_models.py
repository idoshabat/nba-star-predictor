from pathlib import Path
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helper import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directory
from model_settings import (
    FEATURE_COLUMNS,
    MIN_GAMES,
    RANDOM_STATE,
    TARGET_COLUMN,
    XGBOOST_PARAMS,
    XGBOOST_THRESHOLD,
)


os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".matplotlib"))
(REPORTS_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt


DATA_PATH = PROCESSED_DATA_DIR / "nba_allstar_features.csv"
RESULTS_PATH = REPORTS_DIR / "model_comparison.csv"
F1_FIGURE_PATH = FIGURES_DIR / "model_f1_comparison.png"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "xgboost_confusion_matrix.png"

TEST_SIZE = 0.2

MODEL_THRESHOLDS = {
    "Logistic Regression": 0.50,
    "Random Forest": 0.50,
    "XGBoost": XGBOOST_THRESHOLD,
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["gp"] >= MIN_GAMES].copy()
    return df


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(**XGBOOST_PARAMS),
    }


def evaluate_model(name, model, X, y, X_train, X_test, y_train, y_test) -> dict:
    threshold = MODEL_THRESHOLDS[name]
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def save_f1_chart(results: pd.DataFrame) -> None:
    ordered = results.sort_values("f1", ascending=True)

    plt.figure(figsize=(8, 4.5))
    plt.barh(ordered["model"], ordered["f1"], color=["#8da0cb", "#66c2a5", "#fc8d62"])
    plt.xlabel("F1 score")
    plt.title("Model Comparison by F1 Score")
    plt.xlim(0, max(0.7, ordered["f1"].max() + 0.05))

    for index, value in enumerate(ordered["f1"]):
        plt.text(value + 0.01, index, f"{value:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(F1_FIGURE_PATH, dpi=160)
    plt.close()


def save_xgboost_confusion_matrix(results: pd.DataFrame) -> None:
    row = results.loc[results["model"] == "XGBoost"].iloc[0]
    matrix = [
        [row["true_negatives"], row["false_positives"]],
        [row["false_negatives"], row["true_positives"]],
    ]

    plt.figure(figsize=(5, 4.5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("XGBoost Confusion Matrix")
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.colorbar()

    for y_index, values in enumerate(matrix):
        for x_index, value in enumerate(values):
            plt.text(x_index, y_index, int(value), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()


def main() -> None:
    ensure_directory(REPORTS_DIR)
    ensure_directory(FIGURES_DIR)

    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rows = [
        evaluate_model(name, model, X, y, X_train, X_test, y_train, y_test)
        for name, model in build_models().items()
    ]
    results = pd.DataFrame(rows).sort_values("f1", ascending=False)

    results.to_csv(RESULTS_PATH, index=False)
    save_f1_chart(results)
    save_xgboost_confusion_matrix(results)

    print(f"Evaluated {len(results)} models on {len(df):,} rows.")
    print(f"Saved comparison to: {RESULTS_PATH}")
    print(f"Saved F1 chart to: {F1_FIGURE_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
