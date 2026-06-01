# Current State
- Features: CLIP ViT-L/14 + DINOv2 vit_small_patch14 + Gram Matrix (VGG16) → PCA 256-dim
- Best result: ARI=0.240, NMI=0.738 using Leiden (res=70, k=20, threshold=0.4)
- Problem: Each ground truth class splits into 14-36 small clusters, purity 0.12-0.29
- 195 singleton clusters, largest cluster only 41 images while largest class has 94 images

---

## Part 1: Dataset Difficulty Analysis

*   **Composition**: ImageNet-Hard is constructed from six out-of-distribution (OOD) benchmarks: IN-V2, IN-ReaL, IN-A (natural adversarials), IN-R (renditions), IN-S (sketches), and ON (ObjectNet), plus a subset of IN-C (corruptions) (Section 4.4.1). Each sub-dataset contributes unique difficulty: IN-R and IN-S introduce extreme texture variations, IN-A introduces unusual forms, ON introduces cluttered backgrounds and poses, and IN-C introduces severe noise (Impulse Noise, Frost, Fog, Snow, Brightness, Zoom Blur).
*   **Intra-class diversity**: The dataset explicitly tests the ability to recognize abstract images, objects in unusual forms, and unusual poses (Section 4.2). The paper notes that these properties cause "Rare" errors, which are defined as model confusion between semantically distant classes (e.g., `llama` $\rightarrow$ `plectrum`) (Section 4.5, Figure 5b).
*   **Inter-class similarity**: The paper shows models frequently confuse highly related classes, categorized as "Common" errors (which make up 39.4% of EfficientNet-L2's errors). For example, confusing `bucket` with `barrel`, or `clownfish` with `rock beauty` (Section 4.5, Figure 5a).
*   **Cross-domain variance**: The dataset contains real photos, drawings, tattoos, and cartoons within the same label due to the inclusion of IN-R and IN-S. The paper notes that zooming out is particularly important for these abstract images in IN-R and IN-S to perceive the object's shape over its texture (Appendix B.1, Table A2).
*   **Label noise**: The paper estimates a 14.7% labeling noise rate inherited from the source datasets via manual inspection (Section 5, Line 883). Label noise artificially caps clustering metrics like ARI because the ground truth labels themselves are flawed or debatable (Section 4.5).
*   **Resolution and aspect ratio variance**: The paper explicitly states that ObjectNet images were captured using smartphones with aspect ratios of 2:3 or 9:16 (Section 4.2, Line 571). To handle this, the paper's zooming procedure only utilizes `resize` and `crop` to maintain the original aspect ratio (Section 2, Lines 151-152). *The paper does not address the specific minimum and maximum image sizes (70px to 6586px) you observed*.
*   **Why ARI specifically is a misleading metric for this dataset**: *The paper does not address ARI or unsupervised clustering metrics*. However, mathematically, ARI heavily penalizes splitting a ground truth class into multiple clusters. Given the 14.7% label noise (Section 5), "Many objects" images (Section 4.1, Figure 2d), and massive cross-domain variance (sketches vs. photos), clustering algorithms will naturally isolate these domains into separate clusters. ARI interprets this correct visual disentanglement as a "failure" against the single noisy semantic label.

---

## Part 2: Why My Current Pipeline Struggles

*   **Why CLIP ViT-L/14 is the wrong backbone**: The paper specifically constructed ImageNet-Hard by collecting images that OpenAI's CLIP ViT-L/14 misclassifies 100% of the time, even when allowed 324 different zoom attempts (Section 4.4.1). By using CLIP ViT-L/14, you are trying to cluster the exact adversarial subset designed to defeat this model (Table 3 shows CLIP-ViT-L/14@224px gets only 1.86% accuracy). Furthermore, the paper notes CLIP struggles with tightly cropped images because it heavily relies on background context (Section 4.2, Line 581).
*   **Why Gram Matrix actively hurts clustering quality**: The paper emphasizes that ImageNet-Hard contains abstract images (IN-R) and drawings (IN-S) (Section 4.4.1). Because Gram Matrix features capture texture and style, they force the clustering algorithm to group by domain (e.g., all sketches together) rather than semantic class. This directly causes your problem of ground truth classes splitting into 14-36 small clusters based on style rather than content.
*   **Why standard CenterCrop preprocessing fails**: The paper reveals that IN-A and ON have a severe center bias, but standard pre-processing (resize to 256, center crop to 224) is "not optimal for every OOD dataset, not allowing a model to fully utilize off-center visual cues" (Section 4.2). Furthermore, ObjectNet images have smartphone aspect ratios (Line 571), meaning standard square cropping often misses the main object entirely.
*   **Why single-pass feature extraction is insufficient**: The paper shows that test-time augmentation (TTA) using zooming is crucial to accuracy. Using a single pass ignores the dataset's scale variation. The paper proves that aggregating predictions over 16 zoom-in versions of the input image (MEMO + RRC) significantly improves accuracy across all subsets (Section 4.3, Table 2).

---

## Part 3: Implementation Plan

1.  **Backbone replacement**
    *   *What to change*: Replace CLIP ViT-L/14 and Gram Matrix with `EfficientNet-L2@800px` or `EfficientNet-B7@600px`.
    *   *Why*: The paper shows `EfficientNet-L2@800px` achieves the highest accuracy on this dataset (39.00%), far exceeding standard 224x224 models and CLIP (Section 4.4.2, Table 3). Higher resolution is critical here.
    *   *Expected impact*: Significant improvement in capturing semantic features for objects in unusual poses or small scales.
    *   *Implementation priority*: High

2.  **Preprocessing**
    *   *What to change*: Implement the paper's aspect-ratio-preserving zoom strategy. Uniformly resize the image so the *smaller* dimension matches a target scale $S$, then use zero-padding if the content is smaller than 224x224 (Section 3, Lines 177-181).
    *   *Why*: The paper explicitly defines this as the optimal way to avoid aspect ratio distortion while framing the object correctly (Section 3).
    *   *Expected impact*: Prevents objects from being squashed or cropped out, drastically improving feature quality for extreme aspect ratio images.
    *   *Implementation priority*: High

3.  **Multi-crop pooling**
    *   *What to change*: Implement the MEMO + RRC (RandomResizedCrop) strategy. Extract features from $K=16$ randomly zoomed-in versions of the image, and mean-pool them into a single embedding vector.
    *   *Why*: The paper demonstrates that test-time aggregation using zoom-in transforms (RRC) forces models to implicitly zoom on regions of interest, improving accuracy across all five subsets (Section 4.3, Table 2, Figure 4).
    *   *Expected impact*: Embeddings will be robust to scale and translation, reducing the number of singleton clusters caused by bad framing.
    *   *Implementation priority*: High

4.  **Feature fusion**
    *   *What to change*: Keep `EfficientNet-L2@800px` (Semantic) and potentially keep DINOv2 (Structural/Objectness). Drop the Gram Matrix (Style) entirely. Concatenate and PCA to 256-dim.
    *   *Why*: The dataset has extreme cross-domain variance (IN-R, IN-S) (Section 4.4.1). Dropping style features ensures clustering is based on semantic objectness, not texture.
    *   *Expected impact*: Classes will no longer fragment into 14-36 sub-clusters based on domain (photo vs. sketch), directly improving purity and ARI.
    *   *Implementation priority*: High

5.  **Graph construction**
    *   *What to change*: Use a Mutual kNN graph (keep edge only if $A \in kNN(B)$ and $B \in kNN(A)$). Set $k$ slightly higher (e.g., $k=30$).
    *   *Why*: *The paper does not address graph construction*. However, to bridge the domain gap (photos vs sketches of the same class) without introducing false positives across classes caused by "Common" confusion errors (Section 4.5), mutual kNN is structurally necessary.
    *   *Expected impact*: Drops noisy inter-class edges caused by "Common" confusion classes, improving cluster cohesion.
    *   *Implementation priority*: Medium

6.  **Clustering**
    *   *What to change*: Continue using Leiden, but drastically decrease the resolution parameter (e.g., $res=1.0$ or lower) to encourage larger clusters.
    *   *Why*: *The paper does not address clustering hyperparameters*. However, given that your largest cluster is currently 41 images while the largest class has 94 images, your resolution parameter ($res=70$ or $res=60$) is vastly over-segmenting the dataset.
    *   *Expected impact*: Reduces the number of clusters closer to the 1,000 ground truth classes, organically improving ARI.
    *   *Implementation priority*: High

7.  **Evaluation**
    *   *What to change*: Evaluate using metrics robust to label noise, or manually inspect high-purity clusters. Do not rely solely on ARI.
    *   *Why*: The paper establishes a 14.7% label noise rate and the presence of multi-label "Many objects" images (Section 4.1, Section 5). *The paper does not address clustering metrics*.
    *   *Expected impact*: Provides a mathematically realistic evaluation of your pipeline's success, acknowledging that a "perfect" ARI of 1.0 is impossible on this dataset.
    *   *Implementation priority*: Medium
