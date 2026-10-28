import torch
from PIL import Image

from app.embedding import embed_image, embed_text, rank


def test_embed_image_shape_and_norm(tmp_path):
    """TC-01"""
    path = tmp_path / "solid.png"
    Image.new("RGB", (224, 224), color=(120, 30, 200)).save(path)

    vec = embed_image(path)

    assert vec.shape == (1, 512)
    assert torch.isclose(vec.norm(dim=-1), torch.tensor(1.0), atol=1e-5).all()


def test_embed_text_shape_and_norm():
    """TC-02"""
    vec = embed_text("a black bag")

    assert vec.shape == (1, 512)
    assert torch.isclose(vec.norm(dim=-1), torch.tensor(1.0), atol=1e-5).all()


def test_rank_sorted_descending_and_k_larger_than_gallery():
    query = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
    gallery = torch.nn.functional.normalize(torch.randn(4, 512), dim=-1)

    result = rank(query, gallery, k=3)

    assert len(result) == 3
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)

    result_all = rank(query, gallery, k=10)
    assert len(result_all) == 4
    scores_all = [s for _, s in result_all]
    assert scores_all == sorted(scores_all, reverse=True)
