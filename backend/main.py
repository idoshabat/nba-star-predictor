from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.examples import EXAMPLES
from backend.player_lookup import predict_active_player, search_active_players
from backend.predictor import AllStarPredictor
from backend.rookie_rankings import DEFAULT_ROOKIE_SEASON, rookie_rankings
from backend.schemas import (
    PlayerExample,
    PlayerPredictionResponse,
    PlayerSearchResult,
    PlayerStats,
    PredictionResponse,
    RookieRankingItem,
)


app = FastAPI(
    title="NBA Future Star Predictor API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):517\d",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = AllStarPredictor()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_name": predictor.model_name,
        "threshold": predictor.threshold,
    }


@app.get("/examples", response_model=list[PlayerExample])
def examples() -> list[PlayerExample]:
    return EXAMPLES


@app.get("/players/search", response_model=list[PlayerSearchResult])
def search_players(
    query: str = Query(..., min_length=2),
    limit: int = Query(default=10, ge=1, le=25),
) -> list[dict]:
    return search_active_players(query=query, limit=limit)


@app.get("/players/{player_id}/prediction", response_model=PlayerPredictionResponse)
def predict_player(
    player_id: int,
    season_mode: Literal["rookie", "latest"] = Query(default="rookie"),
) -> dict:
    try:
        return predict_active_player(
            player_id=player_id,
            predictor=predictor,
            season_mode=season_mode,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/rookies/rankings", response_model=list[RookieRankingItem])
def rank_rookies(
    season: str = Query(default=DEFAULT_ROOKIE_SEASON, pattern=r"^\d{4}-\d{2}$"),
    limit: int = Query(default=5, ge=1, le=20),
    min_games: int = Query(default=20, ge=1, le=82),
) -> list[dict]:
    try:
        return rookie_rankings(
            predictor=predictor,
            season=season,
            limit=limit,
            min_games=min_games,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/predict", response_model=PredictionResponse)
def predict(player_stats: PlayerStats) -> dict:
    return predictor.predict(player_stats.model_dump())
