from pathlib import Path
import sys
import pandas as pd
import time

from nba_api.stats.endpoints import playercareerstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import RAW_DATA_DIR, ensure_directory


RAW = RAW_DATA_DIR


def get_rookie_stats(player_id):
    """
    Get rookie season statistics for a single player.
    """

    try:
        career = playercareerstats.PlayerCareerStats(
            player_id=int(player_id)
        )

        df = career.get_data_frames()[0]

        # No statistics found
        if df.empty:
            return None


        rookie = df.iloc[0]


        return {
            "player_id": player_id,
            "season": rookie["SEASON_ID"],
            "age": rookie["PLAYER_AGE"],
            "gp": rookie["GP"],
            "gs": rookie["GS"],
            "min": rookie["MIN"],

            "pts": rookie["PTS"],
            "reb": rookie["REB"],
            "ast": rookie["AST"],

            "stl": rookie["STL"],
            "blk": rookie["BLK"],

            "fg_pct": rookie["FG_PCT"],
            "fg3_pct": rookie["FG3_PCT"],
            "ft_pct": rookie["FT_PCT"],

            "tov": rookie["TOV"],
            "pf": rookie["PF"]
        }


    except Exception as e:
        print(
            f"Failed for player {player_id}: {e}"
        )

        return None



def fetch_all_rookie_stats():

    ensure_directory(RAW)


    players = pd.read_csv(
        RAW / "players_with_ids.csv"
    )


    # remove players without ids
    players = players.dropna(
        subset=["id"]
    )


    results = []


    total = len(players)


    for index, row in players.iterrows():

        player_id = row["id"]
        player_name = row["Player_name"]


        print(
            f"[{index+1}/{total}] {player_name}"
        )


        stats = get_rookie_stats(
            player_id
        )


        if stats is not None:

            # keep the name
            stats["player_name"] = player_name

            results.append(stats)


        # avoid NBA API rate limit
        time.sleep(0.6)



    df = pd.DataFrame(results)


    df.to_csv(
        RAW / "rookie_stats.csv",
        index=False
    )


    print()
    print(
        f"Saved {len(df)} players"
    )



if __name__ == "__main__":
    fetch_all_rookie_stats()
