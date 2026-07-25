from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import PROCESSED_DATA_DIR


DATA = PROCESSED_DATA_DIR


def load_data():

    return pd.read_csv(
        DATA / "nba_allstar_prediction.csv"
    )


def main():

    df = load_data()


    print("Shape:")
    print(df.shape)


    print("\nMissing values:")
    print(df.isna().sum())


    print("\nStatistics:")
    print(df.describe())


    print("\nClass balance:")
    print(
        df["is_all_star"]
        .value_counts()
    )


if __name__ == "__main__":
    main()
