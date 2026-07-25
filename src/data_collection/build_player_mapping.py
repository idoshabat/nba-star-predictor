from pathlib import Path
import sys
import pandas as pd
import re
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import RAW_DATA_DIR


RAW = RAW_DATA_DIR
NAME_MAPPING_PATH = RAW / "player_name_mapping.csv"


def normalize_name(name):

    name = name.lower()

    name = unicodedata.normalize(
        "NFKD",
        name
    ).encode(
        "ascii",
        "ignore"
    ).decode(
        "utf-8"
    )

    # remove suffixes
    suffixes = [
        "jr",
        "sr",
        "iii",
        "ii",
        "iv"
    ]

    for suffix in suffixes:
        name = name.replace(
            suffix,
            ""
        )

    name = re.sub(
        r"[^\w\s]",
        "",
        name
    )

    name = name.replace(
        " ",
        ""
    )

    return name

def build_mapping():

    # Load Kaggle All-Star players
    allstars = pd.read_csv(
        RAW / "Players.csv"
    )


    # Load NBA API players
    nba_players = pd.read_csv(
        RAW / "nba_players.csv"
    )

    if NAME_MAPPING_PATH.exists():
        name_mapping = pd.read_csv(NAME_MAPPING_PATH)
        allstars = allstars.merge(
            name_mapping,
            on="Player_name",
            how="left",
        )
        allstars["name_for_matching"] = allstars["nba_name"].fillna(
            allstars["Player_name"]
        )
    else:
        allstars["name_for_matching"] = allstars["Player_name"]

    # Create normalized names
    allstars["normalized_name"] = (
        allstars["name_for_matching"]
        .apply(normalize_name)
    )


    nba_players["normalized_name"] = (
        nba_players["full_name"]
        .apply(normalize_name)
    )


    # Merge using normalized name
    merged = allstars.merge(
        nba_players[
            [
                "id",
                "full_name",
                "normalized_name"
            ]
        ],
        on="normalized_name",
        how="left"
    )


    # Print results
    missing = merged[
        merged["id"].isna()
    ]


    print(
        f"Players without ID: {len(missing)}"
    )


    print(
        missing[["Player_name", "name_for_matching"]]
    )


    # Save result
    merged.to_csv(
        RAW / "players_with_ids.csv",
        index=False
    )


if __name__ == "__main__":
    build_mapping()
