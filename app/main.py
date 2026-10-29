import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from app import db, matching
from app.config import MAX_UPLOAD_MB, get_image_dir
from app.embedding import embed_image

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
VALID_STATUSES = {"unclaimed", "claimed"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(lifespan=lifespan)

_image_dir = get_image_dir()
app.mount("/images", StaticFiles(directory=_image_dir), name="images")


class SearchRequest(BaseModel):
    description: str
    k: int = 5
    place: str | None = None


class StatusRequest(BaseModel):
    status: str


def _serialize_item(item: dict) -> dict:
    result = dict(item)
    image_path = result.pop("image_path", "")
    result["image_url"] = f"/images/{Path(image_path).name}"
    return result


@app.post("/found")
async def create_found(
    image: UploadFile = File(...),
    place: str = Form(""),
    note: str = Form(""),
):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid image type. Only JPEG, PNG and WEBP are allowed.",
        )

    data = await image.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Maximum size is {MAX_UPLOAD_MB} MB.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    suffix = ""
    if image.filename:
        candidate = "." + image.filename.rsplit(".", 1)[-1].lower() if "." in image.filename else ""
        if candidate in ALLOWED_SUFFIXES:
            suffix = candidate
    if not suffix:
        suffix = DEFAULT_SUFFIX[image.content_type]

    image_dir = get_image_dir()
    image_path = image_dir / f"{uuid.uuid4().hex}{suffix}"
    image_path.write_bytes(data)

    try:
        with Image.open(image_path) as uploaded:
            uploaded.verify()
        vector = embed_image(image_path).detach().cpu().numpy().astype(np.float32).reshape(-1)
    except Exception:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    item_id = db.insert_item(image_path, vector, place=place, note=note)
    return _serialize_item(db.get_item(item_id))


@app.post("/search")
def search_items(request: SearchRequest):
    return matching.search(request.description, k=max(1, request.k), place=request.place)


@app.get("/items")
def get_items():
    return [_serialize_item(item) for item in db.list_items()]


@app.get("/items/{item_id}")
def get_item(item_id: int):
    item = db.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found.")
    return _serialize_item(item)


@app.patch("/items/{item_id}/status")
def update_status(item_id: int, request: StatusRequest):
    if request.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail="Invalid status. Allowed values are 'unclaimed' or 'claimed'.",
        )
    if not db.set_status(item_id, request.status):
        raise HTTPException(status_code=404, detail="Item not found.")
    return _serialize_item(db.get_item(item_id))
