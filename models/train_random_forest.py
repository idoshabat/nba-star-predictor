from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"


def train():

    df = pd.read_csv(
        DATA / "nba_allstar_features.csv"
    )


    print(df.shape)


    # Same features as Logistic Regression

    # features = [
    #     "age",
    #     "gp",
    #     "ppg",
    #     "rpg",
    #     "apg",
    #     "mpg",
    #     "starter_rate",
    #     "efficiency",
    #     "pts_per_min",
    #     # "impact_score",
    #     "fg_pct",
    #     "fg3_pct",
    #     "ft_pct"
    # ]
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
    "gs"
]


    X = df[features]

    print("X shape:", X.shape)

    y = df["is_all_star"]


    print("\nFeatures:")
    print(X.columns)



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



    model = RandomForestClassifier(

        n_estimators=300,

        # handle imbalance
        class_weight="balanced",

        random_state=42,

        # avoid overfitting
        max_depth=8,

        min_samples_split=10,

        min_samples_leaf=5,

        n_jobs=-1
    )



    model.fit(
        X_train,
        y_train
    )



    predictions = model.predict(
        X_test
    )


    probabilities = model.predict_proba(
        X_test
    )[:,1]



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



    # Feature importance

    importance = pd.DataFrame({

        "feature": features,

        "importance": model.feature_importances_

    })


    importance = importance.sort_values(
        "importance",
        ascending=False
    )


    print("\nFeature Importance")
    print("------------------")

    print(importance)



if __name__ == "__main__":
    train()