from pathlib import Path
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"


def train():

    # ----------------------------
    # Load dataset
    # ----------------------------
    df = pd.read_csv(
        DATA / "nba_allstar_features.csv"
    )

    print(df.shape)

    # ----------------------------
    # Selected features
    # ----------------------------
    features = [
        "age",
        "gp",
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "min",
        "fg_pct",
        "fg3_pct",
        "ft_pct",
        "pts_per_min",
        "efficiency",
        
    ]

    X = df[features]
    y = df["is_all_star"]

    print("X shape:", X.shape)

    # ----------------------------
    # Train / Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain:")
    print(y_train.value_counts())

    print("\nTest:")
    print(y_test.value_counts())

    # ----------------------------
    # XGBoost Model
    # ----------------------------

    
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        scale_pos_weight=3,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    # ----------------------------
    # Train
    # ----------------------------
    model.fit(
        X_train,
        y_train
    )

    # ----------------------------
    # Predictions
    # ----------------------------
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    threshold = 0.45

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ----------------------------
    # Metrics
    # ----------------------------
    print("\nMetrics")
    print("----------------")

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            predictions
        )
    )

    print(
        "Precision:",
        precision_score(
            y_test,
            predictions
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            predictions
        )
    )

    print(
        "F1:",
        f1_score(
            y_test,
            predictions
        )
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_test,
            probabilities
        )
    )

    print("\nConfusion Matrix")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # ----------------------------
    # Feature Importance
    # ----------------------------
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\nFeature Importance")
    print("------------------")
    print(importance)

    # ----------------------------
    # 5-Fold Cross Validation
    # ----------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    print("\nCross Validation (5-Fold)")
    print("-------------------------")
    print("F1 scores:", scores)
    print(f"Mean F1: {scores.mean():.4f}")
    print(f"Std F1 : {scores.std():.4f}")


if __name__ == "__main__":
    train()