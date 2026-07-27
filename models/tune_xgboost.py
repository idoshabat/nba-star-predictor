from pathlib import Path
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"


def tune():

    df = pd.read_csv(
        DATA / "nba_allstar_features.csv"
    )


    features = [

        # basic
        "age",
        "gp",
        "pts",
        "reb",
        "ast",
        "min",

        # defense
        "stl",
        "blk",

        # shooting
        "fg_pct",
        "fg3_pct",
        "ft_pct",

        # engineered
        "pts_per_min",
        # "impact_score",
        "efficiency",
        "starter_rate",

        # rates
        # "ppg",
        # "rpg",
        # "apg",
        # "mpg"
    ]


    X = df[features]
    y = df["is_all_star"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


    param_grid = {

        "n_estimators": [
            200,
            300,
            500,
            700
        ],

        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1
        ],

        "max_depth": [
            3,
            4,
            5,
            6
        ],

        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "min_child_weight": [
            1,
            3,
            5,
            7
        ],

        "gamma": [
            0,
            0.1,
            0.3,
            0.5
        ],

        "scale_pos_weight": [
            1,
            2,
            3,
            5,
            7,
            9.57
        ]
    }


    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_grid,

        n_iter=30,

        scoring="f1",

        cv=cv,

        random_state=42,

        verbose=2,

        n_jobs=-1

    )


    search.fit(
        X_train,
        y_train
    )


    print("\nBest Parameters")
    print("----------------")
    print(search.best_params_)


    print("\nBest Cross Validation F1")
    print("------------------------")
    print(search.best_score_)



    # ===============================
    # Threshold tuning
    # ===============================

    best_model = search.best_estimator_

    import matplotlib.pyplot as plt


    importance = pd.Series(
        best_model.feature_importances_,
        index=features
    )

    importance.sort_values().plot(
        kind="barh",
        figsize=(8,6)
    )

    plt.title(
        "XGBoost Feature Importance"
    )

    plt.tight_layout()

    plt.show()


    probs = best_model.predict_proba(
        X_test
    )[:, 1]


    print("\nThreshold tuning")
    print("----------------")


    best_f1 = 0
    best_threshold = 0


    for threshold in [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
    ]:


        preds = (
            probs >= threshold
        ).astype(int)


        f1 = f1_score(
            y_test,
            preds
        )


        precision = precision_score(
            y_test,
            preds,
            zero_division=0
        )


        recall = recall_score(
            y_test,
            preds,
            zero_division=0
        )


        print(
            f"Threshold={threshold:.2f} | "
            f"F1={f1:.3f} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f}"
        )


        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold



    print("\nBest threshold")
    print("----------------")
    print(best_threshold)


    print("\nBest Test F1")
    print("----------------")
    print(best_f1)



if __name__ == "__main__":
    tune()