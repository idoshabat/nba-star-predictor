import json
from functools import lru_cache
from pathlib import Path

from nba_api.stats.endpoints import leaguedashplayerstats

from backend.predictor import AllStarPredictor


DEFAULT_ROOKIE_SEASON = "2025-26"
ROOKIE_RANKINGS_CACHE_PATH = Path(__file__).with_name("rookie_rankings_cache.json")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value != value:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_to_stats(row) -> dict:
    return {
        "age": _safe_float(row["AGE"]),
        "gp": int(_safe_float(row["GP"])),
        "pts": _safe_float(row["PTS"]),
        "reb": _safe_float(row["REB"]),
        "ast": _safe_float(row["AST"]),
        "stl": _safe_float(row["STL"]),
        "blk": _safe_float(row["BLK"]),
        "min": _safe_float(row["MIN"]),
        "fg_pct": _safe_float(row["FG_PCT"]),
        "fg3_pct": _safe_float(row["FG3_PCT"]),
        "ft_pct": _safe_float(row["FT_PCT"]),
    }


@lru_cache(maxsize=1)
def rookie_rankings_cache() -> dict[str, list[dict]]:
    if not ROOKIE_RANKINGS_CACHE_PATH.exists():
        return {}

    return json.loads(ROOKIE_RANKINGS_CACHE_PATH.read_text(encoding="utf-8"))


def _live_rookie_rows(season: str) -> list[dict]:
    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="Totals",
        player_experience_nullable="Rookie",
        timeout=10,
    )
    df = response.get_data_frames()[0]

    if df.empty:
        raise ValueError(f"No rookie stats found for season={season}")

    return [
        {
            "player_id": int(row["PLAYER_ID"]),
            "player_name": str(row["PLAYER_NAME"]),
            "team_abbreviation": str(row["TEAM_ABBREVIATION"]),
            "season": season,
            "season_mode": "rookie",
            "stats": _row_to_stats(row),
        }
        for _, row in df.iterrows()
    ]


def rookie_rankings(
    predictor: AllStarPredictor,
    season: str = DEFAULT_ROOKIE_SEASON,
    limit: int = 5,
    min_games: int = 20,
) -> list[dict]:
    rookie_rows = rookie_rankings_cache().get(season)

    if rookie_rows is None:
        rookie_rows = _live_rookie_rows(season)

    rookie_rows = [
        player for player in rookie_rows if player["stats"]["gp"] >= min_games
    ]

    if not rookie_rows:
        raise ValueError(
            f"No rookies found for season={season} with min_games={min_games}"
        )

    ranked_players = []

    for player in rookie_rows:
        stats = player["stats"]
        prediction = predictor.predict(stats)
        ranked_players.append(
            {
                **prediction,
                "player_id": player["player_id"],
                "player_name": player["player_name"],
                "team_abbreviation": player["team_abbreviation"],
                "season": player["season"],
                "season_mode": "rookie",
                "actual_all_star": None,
                "actual_all_star_seasons": [],
                "stats": stats,
            }
        )

    ranked_players.sort(key=lambda item: item["probability"], reverse=True)

    return [
        {
            **player,
            "rank": index + 1,
        }
        for index, player in enumerate(ranked_players[:limit])
    ]
