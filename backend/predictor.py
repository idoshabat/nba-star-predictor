from pathlib import Path
import json
import sys

import joblib
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helper import ARTIFACTS_DIR


CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


class AllStarPredictor:
    def __init__(self) -> None:
        self.config = self._load_config()
        self.model = joblib.load(ARTIFACTS_DIR / self.config["model_file"])
        self.feature_columns = self.config["feature_columns"]
        self.threshold = self.config["threshold"]
        self.model_name = self.config["model_name"]

    def _load_config(self) -> dict:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def build_features(self, payload: dict) -> dict:
        features = dict(payload)
        features["pts_per_min"] = safe_divide(features["pts"], features["min"])
        features["efficiency"] = safe_divide(
            features["pts"]
            + features["reb"]
            + features["ast"]
            + features["stl"]
            + features["blk"],
            features["min"],
        )

        return {
            feature: float(features[feature])
            for feature in self.feature_columns
        }

    def predict(self, payload: dict) -> dict:
        features = self.build_features(payload)
        X = pd.DataFrame([features], columns=self.feature_columns)
        probability = float(self.model.predict_proba(X)[:, 1][0])
        is_all_star = probability >= self.threshold

        return {
            "prediction": "Future All-Star" if is_all_star else "Not projected All-Star",
            "probability": probability,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "signals": build_signals(features),
            "features_used": features,
        }


def build_signals(features: dict) -> list[str]:
    signals = []

    if features["pts"] >= 1000:
        signals.append("High rookie scoring volume")
    elif features["pts_per_min"] >= 0.45:
        signals.append("Strong scoring rate")

    if features["min"] >= 1800:
        signals.append("Trusted with major rookie minutes")

    if features["ast"] >= 250:
        signals.append("Strong playmaking production")

    if features["reb"] >= 400:
        signals.append("Strong rebounding production")

    if features["efficiency"] >= 0.75:
        signals.append("Efficient box-score production")

    if features["age"] <= 21:
        signals.append("Young production curve")

    if not signals:
        signals.append("No standout rookie signal crossed the current rule thresholds")

    return signals[:4]
