from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helper import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ensure_directory
from model_settings import (
    FEATURE_COLUMNS,
    MIN_GAMES,
    RANDOM_STATE,
    TARGET_COLUMN,
    XGBOOST_PARAMS,
    XGBOOST_THRESHOLD,
)


DATA_PATH = PROCESSED_DATA_DIR / "nba_allstar_features.csv"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
METRICS_PATH = ARTIFACTS_DIR / "final_model_metrics.json"
TEST_SIZE = 0.2


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df = df[df["gp"] >= MIN_GAMES].copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def build_model() -> XGBClassifier:
    return XGBClassifier(**XGBOOST_PARAMS)


def evaluate_holdout(model, X_test, y_test) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= XGBOOST_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ensure_directory(ARTIFACTS_DIR)

    X, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    holdout_model = build_model()
    holdout_model.fit(X_train, y_train)
    metrics = evaluate_holdout(holdout_model, X_test, y_test)

    final_model = build_model()
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    config = {
        "model_name": "XGBoost",
        "model_file": MODEL_PATH.name,
        "dataset_file": DATA_PATH.name,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": XGBOOST_THRESHOLD,
        "min_games": MIN_GAMES,
        "random_state": RANDOM_STATE,
        "xgboost_params": XGBOOST_PARAMS,
        "training_rows": int(len(X)),
        "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()),
    }

    save_json(CONFIG_PATH, config)
    save_json(METRICS_PATH, metrics)

    print(f"Trained final XGBoost model on {len(X):,} rows.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved config to: {CONFIG_PATH}")
    print(f"Saved holdout metrics to: {METRICS_PATH}")
    print()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
