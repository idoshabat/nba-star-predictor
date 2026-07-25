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
