import pandas as pd
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import add_model_features
from helper import PROCESSED_DATA_DIR


DATA = PROCESSED_DATA_DIR


def create_features():

    df = pd.read_csv(
        DATA / "nba_allstar_prediction.csv"
    )


    df = add_model_features(df)



    output = DATA / "nba_allstar_features.csv"


    df.to_csv(
        output,
        index=False
    )


    print(df.isna().sum())

    print(df.shape)



if __name__ == "__main__":
    create_features()
