from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


DATA = Path("../../data/processed")


def train_model():

    df = pd.read_csv(
        DATA / "nba_allstar_features.csv"
    )


    # Remove non-features

    drop_columns = [
        "player_id",
        "player_name",
        "season"
    ]


    X = df.drop(
        columns=drop_columns + ["is_all_star"]
    )


    y = df["is_all_star"]



    # Train/Test split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    print("Train:")
    print(y_train.value_counts())

    print("\nTest:")
    print(y_test.value_counts())



    # Scaling

    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(
        X_train
    )


    X_test_scaled = scaler.transform(
        X_test
    )



    # Model

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )


    model.fit(
        X_train_scaled,
        y_train
    )


    # Predictions

    predictions = model.predict(
        X_test_scaled
    )


    probabilities = model.predict_proba(
        X_test_scaled
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



if __name__ == "__main__":
    train_model()