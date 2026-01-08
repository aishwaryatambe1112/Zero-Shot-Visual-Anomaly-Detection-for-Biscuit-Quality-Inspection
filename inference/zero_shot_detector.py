import torch
from models.clip_model import encode_image, encode_text

PROMPTS = [
    "a normal biscuit",
    "a broken biscuit",
    "a burnt biscuit",
    "a biscuit with size defect"
]

TEXT_FEATURES = encode_text(PROMPTS)
CONF_THRESHOLD = 0.65

def detect_defect(biscuit_img):
    image_features = encode_image(biscuit_img)

    similarity = (image_features @ TEXT_FEATURES.T).softmax(dim=-1)
    confidence, index = similarity.max(dim=-1)

    if confidence.item() < CONF_THRESHOLD:
        return "unknown", confidence.item()

    return PROMPTS[index.item()], confidence.item()

