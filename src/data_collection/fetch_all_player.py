from pathlib import Path
import sys
import pandas as pd
from nba_api.stats.static import players

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import RAW_DATA_DIR, ensure_directory


OUTPUT = RAW_DATA_DIR


def fetch_players():

    ensure_directory(OUTPUT)

    nba_players = players.get_players()

    df = pd.DataFrame(nba_players)

    df.to_csv(
        OUTPUT / "nba_players.csv",
        index=False
    )

    print(f"Saved {len(df)} NBA players")


if __name__ == "__main__":
    fetch_players()
