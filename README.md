# Image Clustering on ImageNet-Hard using Graph-based Community Detection

## Introduction

This project investigates and evaluates image clustering methods on the **ImageNet-Hard** dataset by combining modern feature extraction models with graph-based community detection algorithms.

The clustering methods explored in this project include:

* Louvain
* Leiden
* Spectral Clustering
* Markov Clustering (MCL)

The feature extraction backbones used are:

* CLIP ViT-H/14
* CLIP ViT-L/14
* DINOv2
* ResNet50
* EfficientNet-B0
* EfficientNet-B7
* Multi-Model Fusion

---

## Objectives

* Build a robust image feature extraction pipeline using state-of-the-art pretrained models.
* Transform feature representations into similarity graphs.
* Evaluate the effectiveness of graph-based clustering algorithms.
* Compare the representation capabilities of different feature extraction models on the challenging ImageNet-Hard dataset.

---

## Dataset

### ImageNet-Hard

* 10,980 images
* 1,000 semantic categories
* Contains significant variations in viewpoint, image quality, visual style, and noise
* Designed to evaluate robustness under out-of-distribution (OOD) conditions

---

## Experimental Pipeline

1. Image preprocessing
2. Feature extraction using pretrained backbones
3. Dimensionality reduction with PCA
4. Mutual k-NN graph construction
5. Graph-based clustering
6. Evaluation using:

   * ARI (Adjusted Rand Index)
   * NMI (Normalized Mutual Information)
   * Homogeneity
   * Purity

---

## Key Results

The best overall performance was achieved using **EfficientNet-B0 features combined with the Leiden algorithm**:

| Metric      | Score  |
| ----------- | ------ |
| ARI         | 0.3333 |
| NMI         | 0.7776 |
| Homogeneity | 0.7712 |
| Purity      | 0.4977 |

These results demonstrate the effectiveness of combining high-quality semantic feature representations with graph-based community detection techniques for challenging image clustering tasks.

---

## Requirements

* Python 3.10+
* PyTorch
* OpenCLIP
* DINOv2
* scikit-learn
* igraph
* leidenalg
* faiss-cpu
* numpy
* pandas
* matplotlib

Installation:

```bash
pip install -r requirements.txt
```

---

## Team Members

| No. | Full Name               | Student ID | Responsibility                                        |
| --- | ----------------------- | ---------- | ----------------------------------------------------- |
| 1   | Nghiem Nguyen Truong An | 2310013    | Report writing and Louvain implementation             |
| 2   | Huynh Huy Hoang         | 2311041    | Report writing and Markov Clustering implementation   |
| 3   | Ngo Quang Tan           | 2313052    | Report writing and Spectral Clustering implementation |
| 4   | Pham Thanh Tri          | 2313621    | Report writing and Leiden implementation              |

---

## Supervisor

Industrial University of Ho Chi Minh City (IUH)

Data Mining Course Project

---

## Source Code

GitHub Repository:

https://github.com/Susantoco/MProject

---

## License

This project was developed for academic and research purposes only.
