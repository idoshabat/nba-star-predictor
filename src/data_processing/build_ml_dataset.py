from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directory


RAW = RAW_DATA_DIR
PROCESSED = PROCESSED_DATA_DIR


def build_dataset():

    ensure_directory(PROCESSED)


    # All NBA rookie statistics
    all_players = pd.read_csv(
        RAW / "all_players_rookie_stats.csv"
    )


    # All-Star players
    allstars = pd.read_csv(
        RAW / "players_with_ids.csv"
    )


    # Keep only IDs
    allstar_ids = set(
        allstars["id"]
    )


    # Create label
    all_players["is_all_star"] = (
        all_players["player_id"]
        .isin(allstar_ids)
        .astype(int)
    )


    print(
        "Class distribution:"
    )

    print(
        all_players["is_all_star"]
        .value_counts()
    )


    # Save
    all_players.to_csv(
        PROCESSED / "nba_allstar_prediction.csv",
        index=False
    )


    print(
        "Saved dataset:"
    )

    print(
        all_players.shape
    )


if __name__ == "__main__":
    build_dataset()
