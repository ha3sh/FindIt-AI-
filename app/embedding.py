import torch, open_clip
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
model.eval()

def embed_image(path):
    img = preprocess(Image.open(path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        v = model.encode_image(img)
    return torch.nn.functional.normalize(v, dim=-1)

def embed_text(text):
    with torch.no_grad():
        t = model.encode_text(tokenizer([text]))
    return torch.nn.functional.normalize(t, dim=-1)

def rank(query_vec, gallery, k=5):  # gallery: (N, 512) tensor
    scores = (gallery @ query_vec.T).squeeze(1)
    top = torch.topk(scores, k=min(k, len(scores)))
    return list(zip(top.indices.tolist(), top.values.tolist()))
