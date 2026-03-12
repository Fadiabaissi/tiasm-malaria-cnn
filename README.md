# 🦠 Malaria Detection Using Convolutional Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://python.org)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://tensorflow.org)
[![Published: IEEE & Springer](https://img.shields.io/badge/Published-IEEE%20%26%20Springer-green.svg)](#publications)

> Part of the **TIASM** (AI-Powered Platform for E-Health) project  
> LINFI Laboratory, Mohamed Khider University of Biskra, Algeria

---

## Overview

This repository contains the full implementation of a **custom Convolutional Neural Network (CNN)** for binary classification of malaria-infected (*parasitized*) vs. healthy (*uninfected*) red blood cell images from microscopic slides.

The pipeline follows this architecture:

```
Dataset
  └─► Data Preprocessing  (resize 224×224, skip corrupted)
        └─► Data Augmentation  (rotation, zoom, flip, shear)
              └─► CNN Initialisation  (custom 3-block architecture)
                    └─► Feature Extraction
                          └─► Training  (K-Fold, K=3, 150 epochs)
                                └─► Fine Tuning  (Optuna hyperparameter search)
                                      └─► Parasite Identification
                                            └─► Performance Metrics
```

**Key results:** ~99% validation accuracy | No overfitting observed | 3 international publications

---

## Dataset

| Source | Images | License |
|--------|--------|---------|
| NIH / LHNCBC Cell Image Library | 27,558 | Public Domain |
| Local Algerian dataset (NIHPT Blida, 7 cities) | 442 | Institutional |
| **Total** | **~28,000** | |

- **Classes:** Parasitized (1) / Uninfected (0) — balanced (14,000 per class)
- **Image size:** 224 × 224 × 3 (RGB)
- **Validation:** 3-Fold Cross-Validation

The local Algerian dataset was collected in-person at the National Institute for Higher Paramedical Training (NIHPT) in Blida, with microscopic slides from 7 Algerian cities, captured in collaboration with paramedical students and professionals.

---

## Model Architecture

```
Input (224×224×3)
  → Conv2D(32,  3×3, ReLU) → MaxPool(2×2)   ← low-level edge detection
  → Conv2D(64,  3×3, ReLU) → MaxPool(2×2)   ← mid-level pattern extraction
  → Conv2D(128, 3×3, ReLU) → MaxPool(2×2)   ← high-level feature recognition
  → Flatten
  → Dense(128, ReLU)
  → Dropout(0.2)                              ← regularisation
  → Dense(1, Sigmoid)                         ← P(parasitized)
```

**Loss function:** Binary Cross-Entropy  
`L(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`

**Why custom CNN over VGG16/VGG19?**  
Transfer learning models pre-trained on ImageNet encode natural image features that differ significantly from microscopic blood-cell morphology. The custom CNN learns domain-specific filters directly, converging faster with higher accuracy on this task.

---

## Results

| Metric | Value |
|--------|-------|
| Best val accuracy | ~99% |
| Mean val accuracy (K=3) | ~99% ± <1% |
| AUC-ROC | >0.99 |
| Training time | ~1h 33min (150 epochs) |
| Hardware | Linux VM, 16GB RAM, CUDA GPU |

**Training curves:** No overfitting — training and validation curves remain closely aligned across all 150 epochs (see `training_curves_kfold.png`).

---

## Installation

```bash
git clone https://github.com/fadia-baissi/tiasm-malaria-cnn.git
cd tiasm-malaria-cnn
pip install -r requirements.txt
```

### requirements.txt
```
tensorflow==2.10.0
keras==2.10.0
optuna==3.0.3
optkeras==1.0.0
scikit-learn==1.1.3
numpy==1.23.4
matplotlib==3.6.2
seaborn==0.12.1
```

---

## Usage

### Option 1 — Google Colab (recommended)
Open `malaria_cnn_detection.ipynb` in Google Colab and mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Update the dataset paths in **Section 3 (Configuration)**:
```python
INFECTED_PATH   = '/content/drive/MyDrive/dataset/Parasitized'
UNINFECTED_PATH = '/content/drive/MyDrive/dataset/Uninfected'
```

### Option 2 — Local / Linux server
```bash
jupyter notebook malaria_cnn_detection.ipynb
```
Update paths in the Configuration cell to point to your local dataset directory.

---

## Repository Structure

```
tiasm-malaria-cnn/
│
├── malaria_cnn_detection.ipynb   # Main notebook — full pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
│
└── outputs/                       # Generated after running notebook
    ├── augmentation_examples.png
    ├── training_curves_kfold.png
    ├── confusion_matrix.png
    ├── misclassified_examples.png
    └── cnn_malaria_best_fold*.h5  # Saved model checkpoints
```

---

## Publications

This work is described in the following peer-reviewed publications:

1. **Baissi, F., Abdelbaki, E., Kahloul, L., Mohammedi, A., & Ammari, A.** (2024).  
   *Paludism Diagnosis Using Deep Learning.*  
   ICEIS 2024. Atlantis Press / Springer.  
   DOI: [10.2991/978-94-6463-496-9_19](https://doi.org/10.2991/978-94-6463-496-9_19)

2. **Baissi, F., Abdelbaki, E., Kahloul, L., Mohammedi, A., & Ammari, A.** (2024).  
   *TIASM-platform: a New AI-Powered Platform for E-Health.*  
   ICCSA 2024. Springer Nature.  
   DOI: [10.1007/978-3-031-90758-6_23](https://doi.org/10.1007/978-3-031-90758-6_23)

3. **Baissi, F., & Ammari, A.** (2024).  
   *Paludism Detection using Convolutional Neural Network.*  
   ISNIB 2024. IEEE.  
   DOI: [10.1109/ISNIB64820.2025.10982943](https://doi.org/10.1109/ISNIB64820.2025.10982943)

---

## About TIASM

TIASM (التياسم) is an AI-powered e-health platform developed at the LINFI Laboratory, University of Biskra. It integrates multiple diagnostic AI models for malaria, pneumonia, and diabetic retinopathy, and has been officially labelled as an innovative project by regional academic and innovation authorities.

- **Platform team:** 20+ AI researchers, 15+ healthcare professionals  
- **Supervisor:** Prof. Laid Kahloul, LINFI Laboratory  
- **Institution:** Mohamed Khider University of Biskra, Algeria

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The NIH/LHNCBC malaria dataset is in the **public domain** (U.S. Government work).  
Pre-trained VGG weights via Keras Applications are under the **MIT License**.  
TensorFlow/Keras are under the **Apache 2.0 License**.

---

## Author

**Fadia Baissi**  
AI Engineer & Applied Researcher  
LINFI Laboratory, University of Biskra, Algeria  
[LinkedIn](https://linkedin.com/in/fadia-baissi-347498240) · [ResearchGate](https://researchgate.net/profile/Fadia-Baissi)
