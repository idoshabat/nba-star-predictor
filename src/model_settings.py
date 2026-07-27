"""Shared model configuration for training, evaluation, and serving."""


TARGET_COLUMN = "is_all_star"
MIN_GAMES = 20
RANDOM_STATE = 42
XGBOOST_THRESHOLD = 0.45

FEATURE_COLUMNS = [
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

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "scale_pos_weight": 3,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}
