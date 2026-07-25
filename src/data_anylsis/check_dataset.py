from pathlib import Path
import pandas as pd
import numpy as np


DATA = Path("../../data/processed")


def check_dataset():

    df = pd.read_csv(
        DATA / "nba_allstar_features.csv"
    )


    print("=" * 40)
    print("Dataset shape")
    print("=" * 40)

    print(df.shape)



    print("\n" + "=" * 40)
    print("Missing values")
    print("=" * 40)

    missing = df.isna().sum()

    print(
        missing[missing > 0]
    )



    print("\n" + "=" * 40)
    print("Infinite values")
    print("=" * 40)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    print(
        np.isinf(numeric_df).sum()
    )



    print("\n" + "=" * 40)
    print("Duplicates")
    print("=" * 40)

    print(
        df.duplicated().sum()
    )



    print("\n" + "=" * 40)
    print("Data types")
    print("=" * 40)

    print(
        df.dtypes
    )



    print("\n" + "=" * 40)
    print("Target distribution")
    print("=" * 40)

    print(
        df["is_all_star"]
        .value_counts()
    )



    print("\n" + "=" * 40)
    print("Numeric summary")
    print("=" * 40)

    print(
        df.describe()
    )



if __name__ == "__main__":
    check_dataset()