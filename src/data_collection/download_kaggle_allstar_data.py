from pathlib import Path
import sys
import shutil
import kagglehub

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import RAW_DATA_DIR, ensure_directory

DATASET = "ahmedbendaly/nba-all-star-game-data"
OUTPUT_DIR = RAW_DATA_DIR


def download_players_dataset():
    """
    Download the Kaggle dataset and copy only Players.csv
    into data/raw/.
    """

    # Download (or use cached version)
    dataset_path = Path(kagglehub.dataset_download(DATASET))

    # Create output directory
    ensure_directory(OUTPUT_DIR)

    source = dataset_path / "Players.csv"
    destination = OUTPUT_DIR / "Players.csv"

    shutil.copy2(source, destination)

    print(f"Saved {destination}")


if __name__ == "__main__":
    download_players_dataset()
