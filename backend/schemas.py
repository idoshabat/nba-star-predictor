from pydantic import BaseModel, Field


class PlayerStats(BaseModel):
    age: float = Field(..., ge=18, le=50)
    gp: int = Field(..., ge=1, le=82)
    pts: float = Field(..., ge=0)
    reb: float = Field(..., ge=0)
    ast: float = Field(..., ge=0)
    stl: float = Field(..., ge=0)
    blk: float = Field(..., ge=0)
    min: float = Field(..., ge=0)
    fg_pct: float = Field(..., ge=0, le=1)
    fg3_pct: float = Field(..., ge=0, le=1)
    ft_pct: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    threshold: float
    model_name: str
    signals: list[str]
    features_used: dict[str, float]
