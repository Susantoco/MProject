from datasets import load_from_disk
import torch
import numpy as np
import open_clip
import timm

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import os

from utils.utils import load_or_extract, make_loader, preprocessing

ds = load_from_disk("data/imagenet-hard")

images = ds["validation"]["image"]
labels = ds["validation"]["label"]

device = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "../models"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE_CLIP = 32
BATCH_SIZE_DINO = 16

train_tf = preprocessing()

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14',
    pretrained='laion2b_s32b_b79k'
)

clip_feats = load_or_extract(
    "clip_H_14",
    lambda: clip_model,
    make_loader(train_tf, BATCH_SIZE_CLIP, images),
    lambda m, x: m.encode_image(x)
)

del clip_model
torch.cuda.empty_cache()

dino_feats = load_or_extract(
    "dino",
    lambda: timm.create_model(
        'vit_small_patch14_dinov2',
        pretrained=True,
        num_classes=0
    ),
    make_loader(train_tf, BATCH_SIZE_DINO, images),
    lambda m, x: m(x)
)

fused = normalize(np.concatenate([
    1.0 * clip_feats,
    0.7 * dino_feats,
], axis=1))

pca = PCA(whiten=True, random_state=42)
fused = pca.fit_transform(fused)
fused = normalize(fused)

print(f"Final features: {fused.shape}")

np.save(f"{SAVE_DIR}/multi_model_1.npy", fused)
print("Saved!")