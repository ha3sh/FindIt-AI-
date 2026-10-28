import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import db
from app.main import app


def _png_bytes(color=(30, 120, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FINDIT_DB", str(tmp_path / "findit.db"))
    monkeypatch.setenv("FINDIT_IMAGES", str(tmp_path / "images"))
    db.init_db()
    with TestClient(app) as test_client:
        yield test_client


def _post_found(client, **overrides):
    payload = {
        "files": {"image": ("found.png", _png_bytes(), "image/png")},
        "data": {"place": "Library", "note": "black bag"},
    }
    payload.update(overrides)
    return client.post("/found", **payload)


def test_found_then_search_returns_item(client):
    """TC-03"""
    found = _post_found(client)
    assert found.status_code == 200
    item_id = found.json()["item_id"]

    response = client.post("/search", json={"description": "a black bag", "k": 5})

    assert response.status_code == 200
    results = response.json()
    assert [result["item_id"] for result in results] == [item_id]


def test_search_empty_db_returns_empty_list(client):
    """TC-05"""
    response = client.post("/search", json={"description": "a black bag", "k": 5})

    assert response.status_code == 200
    assert response.json() == []


def test_txt_upload_returns_400(client):
    """TC-06"""
    response = client.post(
        "/found",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
        data={"place": "Cafe", "note": "text file"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]


def test_claimed_item_excluded_from_search(client):
    item_id = _post_found(client).json()["item_id"]

    patched = client.patch(f"/items/{item_id}/status", json={"status": "claimed"})
    assert patched.status_code == 200
    assert patched.json()["status"] == "claimed"

    response = client.post("/search", json={"description": "a black bag", "k": 5})
    assert response.status_code == 200
    assert response.json() == []
