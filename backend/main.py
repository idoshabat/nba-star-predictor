from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.examples import EXAMPLES
from backend.predictor import AllStarPredictor
from backend.schemas import PlayerExample, PlayerStats, PredictionResponse


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


@app.post("/predict", response_model=PredictionResponse)
def predict(player_stats: PlayerStats) -> dict:
    return predictor.predict(player_stats.model_dump())
