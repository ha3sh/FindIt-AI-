import json
import sqlite3
from contextlib import contextmanager

import numpy as np

from app.config import get_db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    attributes TEXT,
    place TEXT,
    note TEXT,
    found_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'unclaimed'
)
"""


@contextmanager
def connect():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.execute(_SCHEMA)


def embedding_to_blob(embedding) -> bytes:
    return np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _row_to_item(row: sqlite3.Row, include_embedding: bool = False) -> dict:
    item = dict(row)
    if include_embedding:
        item["embedding"] = blob_to_embedding(item["embedding"])
    else:
        item.pop("embedding", None)
    return item


def insert_item(image_path, embedding, place="", note="", attributes=None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO items (image_path, embedding, attributes, place, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                str(image_path),
                embedding_to_blob(embedding),
                json.dumps(attributes or {}),
                place,
                note,
            ),
        )
        return cur.lastrowid


def get_item(item_id: int):
    with connect() as conn:
        row = conn.execute("SELECT * FROM items WHERE item_id = ?", (item_id,)).fetchone()
        return _row_to_item(row) if row else None


def list_items(status=None, include_embedding: bool = False) -> list[dict]:
    query = "SELECT * FROM items"
    params = ()
    if status is not None:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY item_id"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [_row_to_item(row, include_embedding=include_embedding) for row in rows]


def set_status(item_id: int, status: str) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "UPDATE items SET status = ? WHERE item_id = ?", (status, item_id)
        )
        return cur.rowcount > 0
