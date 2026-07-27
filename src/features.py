"""Shared feature engineering logic for training and serving."""

import pandas as pd


ZERO_FILL_COLUMNS = [
    "gs",
    "fg3_pct",
    "stl",
    "blk",
    "tov",
    "reb",
]

MEDIAN_FILL_COLUMNS = [
    "min",
    "fg_pct",
    "ft_pct",
]


def safe_divide(numerator, denominator):
    result = numerator / denominator

    if isinstance(result, pd.Series):
        return result.replace([float("inf"), -float("inf")], 0).fillna(0)

    if denominator == 0:
        return 0.0

    return result


def clean_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in ZERO_FILL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    for column in MEDIAN_FILL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].median())

    return df


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_player_stats(df)

    df["ppg"] = safe_divide(df["pts"], df["gp"])
    df["rpg"] = safe_divide(df["reb"], df["gp"])
    df["apg"] = safe_divide(df["ast"], df["gp"])
    df["mpg"] = safe_divide(df["min"], df["gp"])

    if "gs" in df.columns:
        df["starter_rate"] = safe_divide(df["gs"], df["gp"])

    df["efficiency"] = safe_divide(
        df["pts"] + df["reb"] + df["ast"] + df["stl"] + df["blk"],
        df["min"],
    )
    df["pts_per_min"] = safe_divide(df["pts"], df["min"])
    df["impact_score"] = (
        df["ppg"] * 0.5
        + df["rpg"] * 0.3
        + df["apg"] * 0.2
    )

    return df.replace([float("inf"), -float("inf")], 0).fillna(0)


def build_prediction_features(payload: dict, feature_columns: list[str]) -> dict:
    df = pd.DataFrame([payload])
    df = add_model_features(df)

    return {
        feature: float(df.loc[0, feature])
        for feature in feature_columns
    }
