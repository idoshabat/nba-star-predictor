from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import PROCESSED_DATA_DIR


DATA = PROCESSED_DATA_DIR


df = pd.read_csv(
    DATA / "nba_allstar_features.csv"
)


features = [
    "age",
    "gp",
    "ppg",
    "rpg",
    "apg",
    "mpg",
    "starter_rate",
    "efficiency",
    "pts_per_min",
    "impact_score",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "is_all_star"
]


corr = df[features].corr()


plt.figure(figsize=(12,10))

plt.imshow(corr)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title(
    "Feature Correlation"
)

plt.tight_layout()

plt.show()
