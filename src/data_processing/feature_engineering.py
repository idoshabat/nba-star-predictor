import pandas as pd
from pathlib import Path


DATA = Path("../../data/processed")


def create_features():

    df = pd.read_csv(
        DATA / "nba_allstar_prediction.csv"
    )


    # =========================
    # Handle missing values
    # =========================


    zero_columns = [
        "gs",
        "fg3_pct",
        "stl",
        "blk",
        "tov",
        "reb"
    ]


    for col in zero_columns:
        df[col] = df[col].fillna(0)



    median_columns = [
        "min",
        "fg_pct",
        "ft_pct"
    ]


    for col in median_columns:

        df[col] = (
            df[col]
            .fillna(df[col].median())
        )



    # =========================
    # Feature engineering
    # =========================


    df["ppg"] = (
        df["pts"] / df["gp"]
    )


    df["rpg"] = (
        df["reb"] / df["gp"]
    )


    df["apg"] = (
        df["ast"] / df["gp"]
    )


    df["mpg"] = (
        df["min"] / df["gp"]
    )


    df["starter_rate"] = (
        df["gs"] / df["gp"]
    )


    df["efficiency"] = (
        df["pts"]
        +
        df["reb"]
        +
        df["ast"]
        +
        df["stl"]
        +
        df["blk"]
    ) / df["min"]

    df["pts_per_min"] = (
    df["pts"] / df["min"]
)

    df["impact_score"] = (
        df["ppg"] * 0.5 +
        df["rpg"] * 0.3 +
        df["apg"] * 0.2
    )


    # replace infinity
    df = df.replace(
        [float("inf"), -float("inf")],
        0
    )



    # remove remaining missing values

    df = df.fillna(0)



    output = DATA / "nba_allstar_features.csv"


    df.to_csv(
        output,
        index=False
    )


    print(df.isna().sum())

    print(df.shape)



if __name__ == "__main__":
    create_features()