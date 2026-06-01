import os
import gc

import numpy as np
import torch
import open_clip
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils.preprocess import LetterboxResize, safe_convert_to_rgb, remove_borders, extract_edge_view


class PipelineConfig:
    IMAGE_SIZE = 224
    PCA_COMPONENTS = 256
    PCA_FIT_SAMPLES = 5000
    CLIP_MODEL_NAME = "ViT-L-14"
    CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"
    DINO_MODEL_NAME = "dinov2_vitb14"
    SEMANTIC_CLASS = "fish"
    
    TEXT_PROMPTS = [
        f"a {SEMANTIC_CLASS}",
        f"a drawing of a {SEMANTIC_CLASS}",
        f"a tattoo of a {SEMANTIC_CLASS}",
        f"a 3D model of a {SEMANTIC_CLASS}",
        f"a photo of a {SEMANTIC_CLASS}",
        f"a sketch of a {SEMANTIC_CLASS}"
    ]

class MultiViewDataset(Dataset):
    def __init__(self, pil_images, clip_transform):
        """
        Takes a list of raw PIL images and transforms them into 3 views.
        Because CLIP and DINOv2 require different normalization, we return both for each view.
        """
        self.images = pil_images
        
        self.letterbox = LetterboxResize(PipelineConfig.IMAGE_SIZE, fill_color=(0,0,0))
        
        self.strong_aug = transforms.Compose([
            transforms.RandomResizedCrop(PipelineConfig.IMAGE_SIZE, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ToTensor(),
        ])
        
        self.clean_base = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.clip_norm = None
        for t in clip_transform.transforms:
            if isinstance(t, transforms.Normalize):
                self.clip_norm = t
                break
        if self.clip_norm is None:
            self.clip_norm = transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
            
        self.dino_norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        raw_img = self.images[idx]
        
        img_rgb = safe_convert_to_rgb(raw_img)
        img_cropped = remove_borders(img_rgb, min_area_ratio=0.6)
        
        img_clean = self.letterbox(img_cropped)
        t_clean = self.clean_base(img_clean)
        
        t_aug = self.strong_aug(img_clean)
        
        w, h = img_cropped.size
        scale = min(self.letterbox.size / w, self.letterbox.size / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img_small = img_cropped.resize((new_w, new_h), Image.BICUBIC)
        
        img_edge = extract_edge_view(img_small)
        img_edge_lb = Image.new('RGB', (self.letterbox.size, self.letterbox.size), (0, 0, 0))
        paste_x = (self.letterbox.size - new_w) // 2
        paste_y = (self.letterbox.size - new_h) // 2
        img_edge_lb.paste(img_edge, (paste_x, paste_y))
        
        t_edge = self.clean_base(img_edge_lb)
        
        return {
            'clip': {
                'clean': self.clip_norm(t_clean),
                'aug': self.clip_norm(t_aug),
                'edge': self.clip_norm(t_edge)
            },
            'dino': {
                'clean': self.dino_norm(t_clean),
                'aug': self.dino_norm(t_aug),
                'edge': self.dino_norm(t_edge)
            }
        }
class LazyImageDataset:
    def __init__(self, hf_ds):
        self.hf_ds = hf_ds
        
    def __len__(self):
        return len(self.hf_ds)
        
    def __getitem__(self, idx):
        return self.hf_ds[idx]["image"]
    
class FeatureExtractionPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.setup_models()
        self.setup_text_features()

    def setup_models(self):
        print("Loading CLIP model...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            PipelineConfig.CLIP_MODEL_NAME, 
            pretrained=PipelineConfig.CLIP_PRETRAINED,
            device=self.device
        )
        self.clip_model.eval()
        
        print("Loading DINOv2 model...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', PipelineConfig.DINO_MODEL_NAME)
        self.dino_model = self.dino_model.to(self.device)
        self.dino_model.eval()

    def setup_text_features(self):
        print("Encoding text prompts...")
        tokenizer = open_clip.get_tokenizer(PipelineConfig.CLIP_MODEL_NAME)
        text_tokens = tokenizer(PipelineConfig.TEXT_PROMPTS).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            self.text_features = F.normalize(text_features, p=2, dim=-1) # Shape: [num_prompts, D_clip]
            
    @torch.no_grad()
    def extract_and_fuse(self, dataloader):
        all_fused_features = []
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            clip_clean = batch['clip']['clean'].to(self.device)
            clip_aug = batch['clip']['aug'].to(self.device)
            clip_edge = batch['clip']['edge'].to(self.device)
            
            dino_clean = batch['dino']['clean'].to(self.device)
            dino_aug = batch['dino']['aug'].to(self.device)
            dino_edge = batch['dino']['edge'].to(self.device)
            
            f_clip_clean = self.clip_model.encode_image(clip_clean)
            f_clip_aug = self.clip_model.encode_image(clip_aug)
            f_clip_edge = self.clip_model.encode_image(clip_edge)
            
            clip_agg = (f_clip_clean + f_clip_aug + f_clip_edge) / 3.0
            clip_agg = F.normalize(clip_agg, p=2, dim=-1)
            
            f_dino_clean = self.dino_model(dino_clean)
            f_dino_aug = self.dino_model(dino_aug)
            f_dino_edge = self.dino_model(dino_edge)
            
            dino_agg = (f_dino_clean + f_dino_aug + f_dino_edge) / 3.0

            dino_agg = F.normalize(dino_agg, p=2, dim=-1)
            text_sim = clip_agg @ self.text_features.T
            
            fused = torch.cat([
                0.7 * clip_agg,
                0.5 * dino_agg,
                text_sim
            ], dim=-1)
            
            all_fused_features.append(fused.cpu().numpy())
            torch.cuda.empty_cache()
            
        return np.vstack(all_fused_features)

    def fit_transform_pca(self, features):
        print(f"Fitting PCA with whitening (components={PipelineConfig.PCA_COMPONENTS})...")
        pca_dim = min(PipelineConfig.PCA_COMPONENTS, features.shape[0], features.shape[1])
        pca = PCA(n_components=pca_dim, whiten=True)
        
        # Fit on subset if dataset is large
        if len(features) > PipelineConfig.PCA_FIT_SAMPLES:
            np.random.seed(42)
            subset_indices = np.random.choice(len(features), PipelineConfig.PCA_FIT_SAMPLES, replace=False)
            pca.fit(features[subset_indices])
        else:
            pca.fit(features)
            
        print("Transforming all features...")
        reduced_features = pca.transform(features)
        
        # Final L2 Normalization
        reduced_features = reduced_features / np.linalg.norm(reduced_features, axis=1, keepdims=True)
        return reduced_features

def run_pipeline(pil_images, batch_size=32, num_workers=4, output_path="features.npy"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    pipeline = FeatureExtractionPipeline(device=device)
    
    dataset = MultiViewDataset(
        pil_images=pil_images,
        clip_transform=pipeline.clip_preprocess
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print("Starting Multi-View Feature Extraction...")
    raw_fused_features = pipeline.extract_and_fuse(dataloader)
    
    del pipeline.clip_model
    del pipeline.dino_model
    torch.cuda.empty_cache()
    gc.collect()
    
    final_embeddings = pipeline.fit_transform_pca(raw_fused_features)
    
    print(f"Saving final features of shape {final_embeddings.shape} to {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, final_embeddings)
    print("Pipeline complete.")
    
    return final_embeddings

if __name__ == "__main__":
    from datasets import load_from_disk
    print("Loading ImageNet-Hard dataset from disk...")
    ds = load_from_disk("data/imagenet-hard")
                
    images = LazyImageDataset(ds["validation"])
    
    print(f"Running pipeline on {len(images)} images...")
    final_features = run_pipeline(images, batch_size=16, num_workers=2, output_path="models/multi_model_2.npy")
    
    print("Successfully generated features! Preview:")
    print(final_features[:2, :5])
