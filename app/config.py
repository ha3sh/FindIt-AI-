import os
from pathlib import Path

ALPHA = 0.7
LOW_CONFIDENCE = 0.2
MAX_UPLOAD_MB = 5


def get_db_path() -> Path:
    path = Path(os.environ.get("FINDIT_DB", "data/findit.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_image_dir() -> Path:
    path = Path(os.environ.get("FINDIT_IMAGES", "data/images"))
    path.mkdir(parents=True, exist_ok=True)
    return path
