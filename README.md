# NBA Future Star Predictor

Machine learning project that predicts whether an NBA player is likely to become a future All-Star based on early-career stats.

## Goal

Given a player's rookie-season profile, predict the probability that the player will become an NBA All-Star later in their career.

## Planned Architecture

1. Data collection from public NBA data sources
2. Dataset creation with rookie stats and future All-Star labels
3. Exploratory data analysis
4. Feature engineering
5. Model training and evaluation
6. SHAP explainability
7. FastAPI prediction backend
8. React dashboard

## Project Structure

```text
nba-star-predictor/
├── backend/
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── create_dataset.py
│   ├── features.py
│   └── train.py
├── README.md
└── requirements.txt
```

## First Milestone

Build a first clean dataset:

```text
player, season, age, ppg, rpg, apg, minutes, fg_pct, three_pct, ft_pct, allstar
```

Then train a baseline model and improve from there.

## Reproducible Pipeline

Run the main project steps from the repository root:

```bash
source venv/bin/activate
python src/data_processing/build_ml_dataset.py
python src/data_processing/feature_engineering.py
python models/evaluate_models.py
python models/train_final_model.py
```

The model comparison report is saved to:

```text
reports/model_comparison.csv
reports/figures/model_f1_comparison.png
reports/figures/xgboost_confusion_matrix.png
```

The final trained model artifacts are saved to:

```text
artifacts/xgboost_model.pkl
artifacts/model_config.json
artifacts/final_model_metrics.json
```

The current evaluation uses players with at least 20 rookie-season games and
compares Logistic Regression, Random Forest, and XGBoost with the same train/test
split and feature set.

## API

Start the FastAPI backend from the repository root:

```bash
source venv/bin/activate
uvicorn backend.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Demo players:

```bash
curl http://127.0.0.1:8000/examples
```

Prediction example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 19,
    "gp": 79,
    "pts": 1654,
    "reb": 432,
    "ast": 465,
    "stl": 130,
    "blk": 58,
    "min": 3122,
    "fg_pct": 0.417,
    "fg3_pct": 0.29,
    "ft_pct": 0.754
  }'
```

The API computes the engineered features required by the model and returns the
All-Star probability, prediction label, threshold, and simple explanatory
signals.

Training and serving share the same feature engineering code in `src/features.py`
so the model receives features consistently in both offline evaluation and API
predictions.

## Frontend

Start the React dashboard in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

The dashboard loads demo players from `/examples`, lets you edit rookie-season
stats, and sends predictions to the FastAPI `/predict` endpoint.
