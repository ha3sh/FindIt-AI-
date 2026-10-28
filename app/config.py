import os
from pathlib import Path

ALPHA = 0.7
LOW_CONFIDENCE = 0.2

DB_PATH = Path(os.environ.get("FINDIT_DB", "data/findit.db"))
IMAGE_DIR = Path("data/images")
MAX_UPLOAD_MB = 5
