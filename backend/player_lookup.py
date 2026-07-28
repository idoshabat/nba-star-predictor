from functools import lru_cache
import json
from pathlib import Path
from typing import Literal

from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players

from backend.actual_outcomes import actual_all_star_for_player, all_star_seasons_for_player
from backend.predictor import AllStarPredictor

SeasonMode = Literal["rookie", "latest"]
ROOKIE_STATS_CACHE_PATH = Path(__file__).with_name("rookie_stats_cache.json")


@lru_cache(maxsize=1)
def active_players() -> list[dict]:
    return [
        {
            "player_id": player["id"],
            "name": player["full_name"],
        }
        for player in players.get_active_players()
    ]


@lru_cache(maxsize=1)
def all_players() -> list[dict]:
    return [
        {
            "player_id": player["id"],
            "name": player["full_name"],
        }
        for player in players.get_players()
    ]


@lru_cache(maxsize=1)
def rookie_stats_cache() -> dict[str, dict]:
    if not ROOKIE_STATS_CACHE_PATH.exists():
        return {}

    return json.loads(ROOKIE_STATS_CACHE_PATH.read_text(encoding="utf-8"))


def search_active_players(query: str, limit: int = 10) -> list[dict]:
    normalized_query = query.strip().lower()

    if not normalized_query:
        return []

    matches = [
        player
        for player in active_players()
        if normalized_query in player["name"].lower()
    ]

    return matches[:limit]


def player_season_stats(player_id: int, season_mode: SeasonMode = "rookie") -> dict:
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=10)
        df = career.get_data_frames()[0]
    except Exception:
        cached_rookie_season = rookie_stats_cache().get(str(player_id))

        if season_mode == "rookie" and cached_rookie_season:
            return cached_rookie_season

        raise

    if df.empty:
        cached_rookie_season = rookie_stats_cache().get(str(player_id))

        if season_mode == "rookie" and cached_rookie_season:
            return cached_rookie_season

        raise ValueError(f"No career stats found for player_id={player_id}")

    if season_mode == "rookie":
        season = df.iloc[0]
    elif season_mode == "latest":
        season = df.iloc[-1]
    else:
        raise ValueError(f"Unsupported season_mode={season_mode}")

    return {
        "season": str(season["SEASON_ID"]),
        "season_mode": season_mode,
        "stats": {
            "age": float(season["PLAYER_AGE"]),
            "gp": int(season["GP"]),
            "pts": float(season["PTS"]),
            "reb": float(season["REB"]),
            "ast": float(season["AST"]),
            "stl": float(season["STL"]),
            "blk": float(season["BLK"]),
            "min": float(season["MIN"]),
            "fg_pct": float(season["FG_PCT"]),
            "fg3_pct": float(season["FG3_PCT"] or 0),
            "ft_pct": float(season["FT_PCT"] or 0),
        },
    }


def predict_active_player(
    player_id: int,
    predictor: AllStarPredictor,
    season_mode: SeasonMode = "rookie",
) -> dict:
    player = next(
        (item for item in all_players() if item["player_id"] == player_id),
        None,
    )

    if player is None:
        raise ValueError(f"Player not found for player_id={player_id}")

    season_payload = player_season_stats(player_id, season_mode=season_mode)
    prediction = predictor.predict(season_payload["stats"])

    return {
        **prediction,
        "player_id": player_id,
        "player_name": player["name"],
        "season": season_payload["season"],
        "season_mode": season_payload["season_mode"],
        "actual_all_star": actual_all_star_for_player(player_id),
        "actual_all_star_seasons": list(all_star_seasons_for_player(player_id)),
        "stats": season_payload["stats"],
    }
