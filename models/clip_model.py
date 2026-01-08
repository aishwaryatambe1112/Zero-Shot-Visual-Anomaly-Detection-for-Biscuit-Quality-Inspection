import torch
import clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def encode_image(img_array):
    image = Image.fromarray(img_array)
    image = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)

    return features

def encode_text(prompts):
    tokens = clip.tokenize(prompts).to(DEVICE)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

    return features

