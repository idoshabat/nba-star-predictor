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
    actual_all_star: bool | None = None
    actual_all_star_seasons: list[str] = Field(default_factory=list)


class PlayerExample(BaseModel):
    player_id: int
    name: str
    stats: PlayerStats
    actual_all_star: bool


class PlayerSearchResult(BaseModel):
    player_id: int
    name: str


class PlayerPredictionResponse(PredictionResponse):
    player_id: int
    player_name: str
    season: str
    season_mode: str
    stats: PlayerStats


class RookieRankingItem(PredictionResponse):
    rank: int
    player_id: int
    player_name: str
    team_abbreviation: str
    season: str
    season_mode: str
    stats: PlayerStats
