import os

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

SEED = 42

# ITEM_COL = "item_id"
# USER_COL = "user_id"
# TIME_COL = "timestamp"
