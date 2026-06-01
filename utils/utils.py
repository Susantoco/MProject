import torch
import numpy as np
import os
import gc
from tqdm import tqdm
from sklearn.preprocessing import normalize
from torchvision import transforms
from .preprocess import ImageDataset
from torch.utils.data import DataLoader

def make_loader(tf, bs, images):
    return DataLoader(
        ImageDataset(images, tf),
        batch_size=bs,
        num_workers=0,
        pin_memory=True
    )

def extract_multi(model, loader, call_fn, name, device='cuda'):
    feats = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=name):
            batch = batch.to(device)
            out1 = call_fn(model, batch)
            out2 = call_fn(model, batch.flip(-1))
            out = (out1 + out2) / 2
            feats.append(out.detach().cpu())
            del out, out1, out2, batch
            torch.cuda.synchronize()

    feats = torch.cat(feats).numpy()
    feats = normalize(feats)
    return feats

def load_or_extract(name, model_fn, loader, call_fn, device='cuda', save_dir='features'):
    path = f"{save_dir}/{name}.npy"

    if os.path.exists(path):
        print(f"{name}: loaded")
        return np.load(path)

    model = model_fn().to(device).eval()

    feats = extract_multi(model, loader, call_fn, name, device)

    np.save(path, feats)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return feats

def preprocessing():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),

        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.7),

        transforms.RandomGrayscale(p=0.3),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23)
        ], p=0.3),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf

def to_single_label(labels):
    if labels.ndim == 1:
        if labels.dtype == object:
            return np.array([l[0] if isinstance(l, (list, tuple, np.ndarray)) else l
                             for l in labels])
        return labels
    return labels.argmax(axis=1)

def load_model(save_dir, feat_file, label_file):
    features = normalize(np.load(f"{save_dir}/{feat_file}"))
    labels   = np.load(f"{save_dir}/{label_file}", allow_pickle=True)
    labels_single = to_single_label(labels)
    return features, labels, labels_single