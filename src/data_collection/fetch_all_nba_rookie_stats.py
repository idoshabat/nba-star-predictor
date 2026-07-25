from pathlib import Path
import sys
import pandas as pd
import time

from nba_api.stats.endpoints import playercareerstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import RAW_DATA_DIR, ensure_directory


RAW = RAW_DATA_DIR

OUTPUT = RAW / "all_players_rookie_stats.csv"


def get_rookie_stats(player_id):

    try:

        career = playercareerstats.PlayerCareerStats(
            player_id=int(player_id)
        )

        df = career.get_data_frames()[0]


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
            f"Failed {player_id}: {e}"
        )

        return None



def fetch_all_players():


    ensure_directory(RAW)


    players = pd.read_csv(
        RAW / "nba_players.csv"
    )


    # only players with ids
    players = players.dropna(
        subset=["id"]
    )


    # resume support
    if OUTPUT.exists():

        old = pd.read_csv(
            OUTPUT
        )

        completed = set(
            old["player_id"]
        )

        results = old.to_dict(
            "records"
        )

    else:

        completed = set()

        results = []


    total = len(players)


    for index, row in players.iterrows():


        player_id = row["id"]


        if player_id in completed:
            continue


        print(
            f"[{index+1}/{total}] {row['full_name']}"
        )


        stats = get_rookie_stats(
            player_id
        )


        if stats:

            stats["player_name"] = row["full_name"]

            results.append(
                stats
            )


        # save every 100 players

        if len(results) % 100 == 0:

            pd.DataFrame(results).to_csv(
                OUTPUT,
                index=False
            )

            print(
                "Progress saved"
            )


        # avoid rate limit

        time.sleep(0.2)



    # final save

    pd.DataFrame(results).to_csv(
        OUTPUT,
        index=False
    )


    print(
        f"Finished. Saved {len(results)} players"
    )



if __name__ == "__main__":
    fetch_all_players()
