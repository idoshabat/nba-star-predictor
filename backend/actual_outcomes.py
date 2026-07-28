import csv
import json
from functools import lru_cache
from pathlib import Path

from nba_api.stats.endpoints import playerawards


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "nba_allstar_prediction.csv"
AWARDS_CACHE_PATH = Path(__file__).with_name("all_star_awards_cache.json")


@lru_cache(maxsize=1)
def all_star_outcomes() -> dict[int, bool]:
    if not LABELS_PATH.exists():
        return {}

    outcomes = {}

    with LABELS_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            player_id = row.get("player_id")
            label = row.get("is_all_star")

            if player_id is None or label is None:
                continue

            outcomes[int(player_id)] = label == "1"

    return outcomes


@lru_cache(maxsize=1)
def all_star_awards_cache() -> dict[str, list[str]]:
    if not AWARDS_CACHE_PATH.exists():
        return {}

    return json.loads(AWARDS_CACHE_PATH.read_text(encoding="utf-8"))


def actual_all_star_for_player(player_id: int) -> bool | None:
    seasons = all_star_seasons_for_player(player_id)

    if seasons:
        return True

    fallback = all_star_outcomes().get(player_id)

    if fallback is True:
        return True

    return False


@lru_cache(maxsize=2048)
def all_star_seasons_for_player(player_id: int) -> tuple[str, ...]:
    cached_seasons = all_star_awards_cache().get(str(player_id))

    if cached_seasons is not None:
        return tuple(cached_seasons)

    try:
        awards = playerawards.PlayerAwards(player_id=player_id, timeout=10).get_data_frames()[0]
    except Exception:
        return ()

    if awards.empty:
        return ()

    descriptions = awards["DESCRIPTION"].fillna("").astype(str)
    subtypes = awards["SUBTYPE1"].fillna("").astype(str)
    all_star_awards = awards[
        descriptions.str.contains("NBA All-Star", case=False, regex=False)
        | subtypes.str.fullmatch("All-Star", case=False)
    ]

    return tuple(sorted(set(all_star_awards["SEASON"].dropna().astype(str))))
