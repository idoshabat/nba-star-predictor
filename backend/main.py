from fastapi import FastAPI

from backend.predictor import AllStarPredictor
from backend.schemas import PlayerStats, PredictionResponse


app = FastAPI(
    title="NBA Future Star Predictor API",
    version="0.1.0",
)

predictor = AllStarPredictor()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_name": predictor.model_name,
        "threshold": predictor.threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(player_stats: PlayerStats) -> dict:
    return predictor.predict(player_stats.model_dump())
