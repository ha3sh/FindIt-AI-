from pathlib import Path

import numpy as np
import torch

from app import db
from app.embedding import embed_text, rank


def search(description: str, k: int = 5) -> list[dict]:
    rows = db.list_items(status="unclaimed", include_embedding=True)
    if not rows:
        return []

    gallery = torch.from_numpy(
        np.stack([row["embedding"] for row in rows]).astype(np.float32)
    )
    query = embed_text(description)

    results = []
    for index, score in rank(query, gallery, k=k):
        row = rows[index]
        results.append(
            {
                "item_id": row["item_id"],
                "score": float(score),
                "image_url": f"/images/{Path(row['image_path']).name}",
                "place": row["place"],
                "note": row["note"],
            }
        )
    return results
