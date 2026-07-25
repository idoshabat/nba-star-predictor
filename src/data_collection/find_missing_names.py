from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import RAW_DATA_DIR


RAW = RAW_DATA_DIR


def find_names():

    merged = pd.read_csv(
        RAW / "players_with_ids.csv"
    )

    nba_players = pd.read_csv(
        RAW / "nba_players.csv"
    )


    missing = merged[
        merged["id"].isna()
    ]


    for name in missing["Player_name"]:
        print("\nSearching:", name)

        matches = nba_players[
            nba_players["full_name"]
            .str.contains(
                name.split()[0],
                case=False,
                na=False
            )
        ]

        print(
            matches["full_name"].tolist()
        )


if __name__ == "__main__":
    find_names()
