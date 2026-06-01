import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx].convert("RGB"))
    

def safe_convert_to_rgb(image: Image.Image) -> Image.Image:
    """Safely convert images to RGB, handling RGBA compositing and other modes."""
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
        return image.convert('RGB')
    elif image.mode != 'RGB':
        return image.convert('RGB')
    return image

def remove_borders(image: Image.Image, min_area_ratio: float = 0.6) -> Image.Image:
    """Automatically remove borders using variance heuristics."""
    gray = np.array(image.convert('L'))
    row_vars = np.var(gray, axis=1)
    col_vars = np.var(gray, axis=0)
    
    var_thresh = 5.0
    row_indices = np.where(row_vars > var_thresh)[0]
    col_indices = np.where(col_vars > var_thresh)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        return image 
        
    top, bottom = row_indices[0], row_indices[-1]
    left, right = col_indices[0], col_indices[-1]
    
    top = max(0, top - 2)
    bottom = min(gray.shape[0], bottom + 2)
    left = max(0, left - 2)
    right = min(gray.shape[1], right + 2)
    
    original_area = gray.shape[0] * gray.shape[1]
    new_area = (bottom - top) * (right - left)
    
    if new_area / original_area >= min_area_ratio:
        return image.crop((left, top, right, bottom))
    
    return image

class LetterboxResize:
    """Pad to square while preserving aspect ratio, then resize."""
    def __init__(self, size=224, fill_color=(0, 0, 0)):
        self.size = size
        self.fill_color = fill_color
        
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        target_w, target_h = self.size, self.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        img = img.resize((new_w, new_h), Image.BICUBIC)
        new_img = Image.new('RGB', (target_w, target_h), self.fill_color)
        
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img

def extract_edge_view(image: Image.Image) -> Image.Image:
    """Apply Canny edge detection and convert to 3-channel RGB image."""
    cv_img = np.array(image.convert('L'))
    
    v = np.median(cv_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(cv_img, lower, upper)

    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_3ch)

