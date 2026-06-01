# ImageNet-Hard Feature Extraction Pipeline: Design Rationale

This document explains the technical rationale behind the `feature_extraction_pipeline.py` script and how its specific design choices directly address the unique challenges posed by the **ImageNet-Hard** dataset.

## 1. The Challenge of ImageNet-Hard
As described in the study, **ImageNet-Hard** consists of images that state-of-the-art models (including CLIP-ViT-L/14) completely fail to classify, *even when allowed 324 optimal zoom attempts*. These images are sourced from challenging datasets containing extreme variations:
- **IN-A & ObjectNet (ON):** Rare poses, unusual backgrounds, and strong center/positional biases.
- **IN-R & IN-Sketch (IN-S):** Abstract renditions, art, and black-and-white sketches.
- **IN-C:** Severe corruptions, blur, noise, and weather artifacts.
- **"Unclassifiable" cases:** Illusions, occlusion, and heavy clutter (many objects).

To extract features from such a dataset, standard 224x224 center-crop pipelines are insufficient. The pipeline utilizes multi-view geometry, dual-foundation models, and semantic grounding to extract robust, semantically invariant, and structurally aware features.

## 2. Aspect-Preserving Preprocessing
**Components:** `remove_borders()` and `LetterboxResize`

> [!NOTE]
> **Why it's used:** Traditional preprocessing (resize and center crop) distorts object aspect ratios and discards off-center features. The paper notes that ImageNet-A and ObjectNet suffer heavily from off-center objects and varying aspect ratios (e.g., 9:16 smartphone photos).

- **Border Removal:** Automatically crops out artificial frames or letterboxing commonly found in web-scraped images or specific datasets, reducing dead space that models might falsely fixate on.
- **Letterbox Resizing:** Pads the image to a square using a solid background rather than warping it. This preserves the original aspect ratio, ensuring that rare object poses (a key failure mode in ImageNet-Hard) are not geometrically deformed before feature extraction.

## 3. Multi-View Representation
**Components:** `MultiViewDataset` (Clean, Strong Augmentation, Edge View)

Because ImageNet-Hard contains heavy domain shifts (sketches, 3D models, corrupted photos), relying on a single RGB view is brittle. The pipeline aggregates features across three distinct views:
1. **Clean View:** The baseline letterboxed image.
2. **Strong Augmentation:** Applies heavy color jitter, grayscale conversion, and Gaussian blur. 
   - *Rationale:* Forces the extracted features to be invariant to color and texture, which is critical for handling **IN-C** (corruptions) and **IN-R** (renditions).
3. **Edge View (Canny Edge Detection):** Extracts pure shape and structural boundaries.
   - *Rationale:* Directly addresses **IN-Sketch (IN-S)** and abstract arts where texture is entirely absent but object shape remains. By feeding a Canny edge map, the model extracts features based purely on geometric contours, bypassing misleading textures.

## 4. Dual Foundation Model Fusion
**Components:** `CLIP (ViT-L-14)` + `DINOv2 (ViT-B14)`

> [!TIP]
> **Why it's used:** Different foundation models possess different inductive biases. By fusing them, the pipeline leverages the strengths of both.

- **CLIP:** Trained via contrastive language-image pretraining on vast amounts of internet data. It possesses robust high-level semantic understanding and is highly resilient to stylistic changes (sketches vs. photos), but it can struggle with precise spatial localization and dense clutter.
- **DINOv2:** A self-supervised vision transformer that excels at pixel-level understanding, identifying object boundaries, and parts-based localization. 
- **Fusion (`0.7 * clip_agg + 0.5 * dino_agg`):** DINOv2 helps separate objects from cluttered backgrounds (solving the "many objects" and "occlusion" failure modes), while CLIP provides the stylistic and semantic invariance needed for diverse renditions.

## 5. Semantic Text Prompting
**Components:** `TEXT_PROMPTS` and `text_sim = clip_agg @ self.text_features.T`

The pipeline explicitly measures the cosine similarity between the image features and a set of predefined textual domain prompts (e.g., *"a drawing of a {class}"*, *"a 3D model of a {class}"*, *"a sketch of a {class}"*).

> [!IMPORTANT]
> **Why it's used:** Instead of relying entirely on visual features, this injects explicit semantic priors into the feature vector. If an image is a sketch, the feature vector will have high similarity in the "sketch" dimension. This explicitly maps the visual domain gaps present in ImageNet-Hard into measurable, structured feature signals, aiding downstream tasks in grouping images by their domain or style.

## 6. PCA Whitening & Normalization
**Components:** `PCA(n_components=256, whiten=True)`

The raw fused tensor concatenates CLIP embeddings, DINOv2 embeddings, and Text similarities into a very high-dimensional vector.
- **Dimensionality mismatch:** CLIP, DINO, and text similarity metrics have vastly different output dimensions and feature variances.
- **Whitening:** Applying PCA with `whiten=True` decorrelates the features and scales them to unit variance. 
- **Outcome:** This ensures that no single modality (e.g., CLIP's larger embedding space) dominates the others. It creates a compact, isotropic 256-dimensional space optimized for distance-based algorithms like K-Means or DBSCAN clustering, ensuring all extracted signals (semantic, spatial, stylistic) contribute equally.

## Summary
This pipeline is explicitly engineered to counteract the failure modes identified in the ImageNet-Hard paper. It preserves geometry via letterboxing, eliminates texture biases using edge and augmented views, combines spatial structure (DINOv2) with semantic robustness (CLIP), and creates a normalized, domain-aware embedding space.
