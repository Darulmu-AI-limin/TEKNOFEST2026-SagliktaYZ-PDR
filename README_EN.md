<div align="center">

# 🫀 EKG-Tarayıcı: 12-Lead ECG Classification & Clinical Decision Support System
### TEKNOFEST 2026 — Artificial Intelligence in Healthcare (High School Category)
#### **Team:** Devre181 | **Team ID:** #987840 | **Application ID:** #5218603

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TEKNOFEST](https://img.shields.io/badge/TEKNOFEST-2026-red.svg)](https://www.teknofest.org/)

**[🇹🇷 Türkçe](READMEcopy.md) | [🇬🇧 English](README_EN.md)**

<p align="center">
  <b>An end-to-end hybrid deep learning clinical decision support system designed to detect arrhythmias and conduction disorders from 12-lead ECG signals, featuring an explainable architecture and an automated rejection mechanism for unfamiliar pathologies ("Safe-Fail / Unknown").</b>
</p>

</div>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Team Structure](#-team-structure)
- [Model Performance & Empirical Results](#-model-performance--empirical-results)
- [Target Classes & Safe-Fail Rejection](#-target-classes--safe-fail-rejection)
- [Model Architecture](#-model-architecture)
- [Elimination of Alternative Architectures](#-elimination-of-alternative-architectures)
- [Datasets & Signal Preprocessing Pipeline](#-datasets--signal-preprocessing-pipeline)
- [Class Imbalance & Data Augmentation](#-class-imbalance--data-augmentation)
- [Data Splitting & Experimental Protocol](#-data-splitting--experimental-protocol)
- [Loss Function (Asymmetric Loss)](#-loss-function-asymmetric-loss)
- [Decision Logic & Calibration](#-decision-logic--calibration)
- [Hyperparameters & Training Settings](#-hyperparameters--training-settings)
- [Technical Evolution & Engineering Decisions](#-technical-evolution--engineering-decisions)
- [Explainability (Attention Rollout)](#-explainability-attention-rollout)
- [Hardware & Performance Benchmarks](#-hardware--performance-benchmarks)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Getting Started](#-installation--getting-started)
- [References & Literature](#-references--literature)
- [License](#-license)

---

## 📖 Overview

**EKG-Tarayıcı** is an end-to-end clinical decision support system engineered by team **Devre181** for the **TEKNOFEST 2026 Artificial Intelligence in Healthcare** competition.

Rather than transforming 1D physiological time-series into cumbersome 2D spectrograms, this project deploys a lightweight **hybrid architecture (ResNet-1D + CBAM + Lead-Transformer)** that operates directly on raw voltage signals.

Beyond strong diagnostic accuracy, the model delivers **sub-second inference latencies** suitable for edge devices and clinical bedside monitors, coupled with a vital safety feature: an automated **Safe-Fail (Unknown)** verdict when confronted with ambiguous or out-of-distribution (OOD) cardiac morphologies.

---

## 📊 Model Performance & Empirical Results

Validation set metrics reported in the official Project Detailed Report (PDR):

| Metric | Score |
|:---|:---:|
| **Overall Accuracy** | **85.3%** |
| **Macro F1-Score** | **0.768** |
| **Macro Precision** | **0.800** |
| **Macro Recall** | **0.800** |

### Per-Class Diagnostic Performance

| Class | Diagnosis | Precision | Recall | F1-Score |
|:---:|:---|:---:|:---:|:---:|
| **NORMAL** | Normal Sinus Rhythm | 0.92 | 0.92 | **0.92** |
| **AFIB** | Atrial Fibrillation | 0.84 | 0.81 | **0.82** |
| **AFL** | Atrial Flutter | 0.83 | 0.81 | **0.82** |
| **RBBB** | Right Bundle Branch Block | 0.72 | 0.73 | **0.73** |
| **LBBB** | Left Bundle Branch Block | 0.69 | 0.74 | **0.71** |
| **Average** | **Macro Average** | **0.80** | **0.80** | **0.768** |

### Confusion Matrix
```
Ground Truth \ Pred   Normal    AFIB     AFL     RBBB    LBBB
Normal                  458      12       6       14      10
AFIB                     18     203      22        4       3
AFL                       9      24     162        3       2
RBBB                      8       2       3      132      35
LBBB                      7       1       2       29     111
```
*Error analysis demonstrates that boundary confusions predominantly occur between morphologically similar conduction defects (LBBB–RBBB) and rhythm abnormalities with close rate characteristics (AFIB–AFL).*

---

## 🎯 Target Classes & Safe-Fail Rejection

The network evaluates 10-second (2500 samples @ 250 Hz) 12-lead records across 5 primary categories:
`NORMAL`, `AFIB`, `AFL`, `RBBB`, `LBBB`.

### 🛡️ Safe-Fail / Unknown Mechanism
Forcing a clinical model to classify unseen pathologies (e.g., Acute Myocardial Infarction [MI], WPW syndrome) into predefined classes creates catastrophic medical risk.
- **Negative Anchoring:** Non-target diagnostic records are assigned the zero target vector `[0, 0, 0, 0, 0]`. The network learns to actively suppress activations on unfamiliar morphologies.
- **OOD Validation:** **4,000 non-target records** (3,000 in pretraining, 1,000 in fine-tuning) were evaluated to verify that the model reliably suppresses confidence on unfamiliar morphologies.
- **Rejection Rule:** Signals failing calibrated confidence thresholds or presenting minimal inter-class margin are routed to **Unknown**, preventing false-positive interventions.

---

## 🏗️ Model Architecture

An end-to-end trainable three-stage hybrid deep neural network:

```
Input: [Batch, 12, 2500] (12 Leads × 2500 Time-steps)
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  ResNet-1D (4 Residual Blocks)                         │
│  - Independent convolutions per lead                   │
│  - Kernel: 7, Stride: 2, MaxPool, Residual skips       │
│  - Output: [Batch * 12, 256, T]                        │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  CBAM 1D (Convolutional Block Attention Module)        │
│  - Channel Attention: Highlights informative leads,    │
│    suppresses noisy/dead electrode channels            │
│  - Temporal Attention: Focuses on QRS/P/T segments     │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  Global Average Pooling (GAP)                          │
│  - Compresses temporal dimension (T)                   │
│  - Output: [Batch, 12, 256]                            │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  Pre-LN Transformer Encoder (2 Layers, 4 Heads)        │
│  - Learnable CLS token appended -> [Batch, 13, 256]    │
│  - Captures 3D spatial cardiac electrical vector       │
│    dynamics across all 12 anatomical leads             │
│  - Output (CLS): [Batch, 256]                          │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  Classification Head                                   │
│  - LayerNorm + Linear(256 -> 5)                        │
│  - Independent Sigmoid activations (Multi-label)       │
└────────────────────────────────────────────────────────┘
```

> **Why Sigmoid instead of Softmax?**  
> Clinical patients can concurrently exhibit both an arrhythmia (e.g., AFIB) and a conduction disorder (e.g., RBBB). Softmax forces probabilities to sum to 1, violating medical realism; independent **Sigmoid** outputs treat each pathology as a separate binary decision.

---

## 🚫 Elimination of Alternative Architectures

| Alternative Architecture | Grounds for Elimination |
|:---|:---|
| **Classical Machine Learning (SVM, Random Forest)** | Required manual feature engineering; failed to encompass the rich morphologic diversity of P-QRS-T complexes and inter-lead spatial vectors. |
| **Pure Transformer Models** | Suffered from weak inductive bias on limited biomedical datasets, resulting in erratic convergence and severe overfitting risk. |
| **Softmax-Based Output Layer** | Enforced mutual exclusivity across classes, which violates clinical reality where rhythm and conduction abnormalities regularly co-occur. |
| **WeightedRandomSampler** | Distorted the true prior probability distribution of clinical data, compromised calibration, and induced acute overconfidence on OOD signals. |

---

## 🔬 Datasets & Signal Preprocessing Pipeline

### Dataset Breakdown (PDR Specifications)

| Dataset | Source | Total Records | Sampling / Length | Purpose | Formats |
|:---|:---|:---:|:---:|:---:|:---|
| **TEKNOFEST** | TEKNOFEST | 5,000 | 500Hz / 10s | Fine-tuning | .mat / .dat, .hea |
| **PTB-XL** | PhysioNet | 21,799 | 500Hz / 10s | Pre-training | .dat, .hea |
| **ECG-Arrhythmia** | PhysioNet (Chapman) | 45,152 | 500Hz / 10s | Pre-training | .mat, .hea |
| **G12EC** | PhysioNet Challenge (Georgia) | 10,344 | 500Hz / 5-10s | Pre-training | .mat, .hea |

*A unified corpus of **41,601 records** was gathered across target classes (NORMAL: 20,388, AFIB: 4,864, AFL: 9,319, RBBB: 4,946, LBBB: 2,084).*

### Signal Pipeline
1. **Bandpass Filtering (0.5 – 45 Hz):** Eliminates respiratory baseline drift and high-frequency EMG artifacts while strictly safeguarding vital QRS complexes.
2. **Notch Filtering (50 Hz):** Attenuates electrical powerline hum.
3. **Lead-Wise Z-Score Normalization:** Standardizes voltage amplitude per lead while preserving flatline/dead channels ($std < 10^{-6}$) to avoid numerical instability.
4. **Quality Screening & Resampling:**
   - Resampled to 250 Hz (10 seconds $\rightarrow$ 2500 points).
   - Excluded incomplete leads and recordings $< 8$ seconds.
   - Removed clipped records ($> 5\%$ of samples at ADC limits) and voltage spikes ($> 10\text{ mV}$).
   - Fingerprint deduplication purged 100 duplicate patient entries.
5. **HDF5 RAM In-Memory Caching:** Slices are stored as compressed `float16` tensors. Direct memory access by PyTorch workers reduced epoch training time from **12 minutes to under 2 minutes**.

---

## ⚖️ Class Imbalance & Data Augmentation

Normal rhythm constituted 49.01% of the pooled dataset. To prevent model collapse into the majority class, an aggressive balancing protocol was deployed:

### Dataset Partitioning and Weight Schedule

| Class | Included Base Records | Post-Augmentation Records | Class Weight |
|:---:|:---:|:---:|:---:|
| **NORMAL** | 18,000 | 18,000 | **0.45** |
| **AFIB** | 4,800 | 9,600 *(x2)* | **1.00** |
| **AFL** | 8,000 | 8,000 | **1.20** |
| **RBBB** | 4,900 | 9,800 *(x2)* | **0.85** |
| **LBBB** | 2,000 | 4,000 *(x2)* | **2.90** |
| **Total** | **37,700** | **49,400** | — |

*Log-scale damping reduced the theoretical inverse frequency weight of LBBB from 4.7 down to 2.9 to maintain gradient stability.*

### Augmentation Suite (Minority Classes in Training Only)
- **Gaussian Noise:** Sensor jitter simulation.
- **Baseline Wander:** 0.1–0.5 Hz breathing oscillation.
- **Amplitude Scaling:** 0.95–1.05 range modulation preserving morphology.
- **Time Shift:** ±300 ms temporal translation.

---

## 🧪 Data Splitting & Experimental Protocol

- **Patient-Wise Partitioning (GroupShuffleSplit):** Enforced strict patient separation to prevent data leakage between training and evaluation splits.
- **Pre-training Partition:** PTB-XL, Chapman, and Georgia data were partitioned into **80% Training, 20% Validation**.
- **Fine-tuning Partition:** TEKNOFEST competition data was split into **70% Training, 15% Validation, 15% Test**.
- **Safe-Fail / Unknown Protocol:** 4,000 out-of-distribution records (3,000 pretraining + 1,000 fine-tuning) were validated to confirm rejection behavior.

---

## 📐 Loss Function (Asymmetric Loss)

To tackle severe class disparity and the extreme risk asymmetry of missing critical conditions, **Asymmetric Loss (ASL)** is employed:

$$\mathcal{L} = \sum_{k=1}^{K} - y_k (1 - p_k)^{\gamma_{pos}} \log(p_k) - (1 - y_k) (p_{m,k})^{\gamma_{neg}} \log(1 - p_{m,k})$$

- $\gamma_{pos} = 0$: Prevents over-penalizing confident correct positive classifications.
- $\gamma_{neg} = 2$: Aggressively penalizes false negatives, maximizing sensitivity for life-threatening diagnoses (e.g., LBBB).
- **Asymmetric Clipping ($clip = 0.05$):** Zeros out clean negatives to channel capacity into difficult borderline examples.

---

## 🚦 Decision Logic & Calibration

1. **Temperature Scaling:** Raw logits are calibrated on validation data via $z / T$ to ensure predicted probabilities match empirical accuracy.
2. **Class-Specific Thresholds:** Grid search over 0.50–0.95 with step 0.01 selects optimal cutoff values per class to maximize Macro F1.
3. **Margin Control & Safe-Fail Rule:**
   ```
   Top Probability < Threshold               ───► [ UNKNOWN ]
   Top Probability >= 0.80                  ───► [ ACCEPT (High Confidence) ]
   Threshold <= Top Probability < 0.80:
      Margin (Top1 - Top2) < Margin_Limit   ───► [ UNKNOWN (Ambiguous / Low Margin) ]
      Margin (Top1 - Top2) >= Margin_Limit  ───► [ ACCEPT (Top1 Class) ]
   ```

---

## ⚙️ Hyperparameters & Training Settings

| Parameter | Pre-training Phase | Fine-tuning Phase |
|:---|:---:|:---:|
| **Optimizer** | AdamW (Weight Decay = $1\times 10^{-4}$) | AdamW (Weight Decay = $1\times 10^{-4}$) |
| **Learning Rate (LR)** | 5 Epoch Warmup $\rightarrow 1\times 10^{-3}$, Cosine Annealing $\rightarrow 1\times 10^{-5}$ | Cosine Scheduler $\rightarrow 2\times 10^{-4}$ |
| **Batch Size** | 64 | 32 |
| **Epochs** | 30 Epochs *(Theoretical schedule: 80–100)* | 20–30 Epochs |
| **Regularization** | 0.2 Dropout, Gradient Clipping (max norm = 1.0) | 0.2 Dropout, Gradient Clipping (max norm = 1.0) |
| **Early Stopping** | Patience = 5 Epochs (Tracking Validation Macro F1) | Patience = 5 Epochs (Tracking Validation Macro F1) |
| **Precision** | Automatic Mixed Precision (AMP - FP16) | Automatic Mixed Precision (AMP - FP16) |

---

## 🔄 Technical Evolution & Engineering Decisions

1. **Stage 1 (Baseline CNN):** A vanilla 1D CNN collapsed into majority classes, obtaining a Macro-F1 of only **~0.59**. This necessitated the ResNet-1D + CBAM + Transformer hybrid backbone.
2. **Stage 2 (Sampler Distortion & Overconfidence):** Applying `WeightedRandomSampler` disrupted natural prior distributions, ruined probability calibration, and caused severe overconfidence on out-of-distribution (OOD) waveforms.
3. **Resolution & Integration:** The sampler was discarded; imbalance handling was reassigned to **ASL + Class-Weights**. In-memory **HDF5 RAM Caching** solved I/O bounds. Integrating the Safe-Fail decision framework elevated Macro-F1 to **0.768**.

---

## 💡 Explainability (Attention Rollout)

For clinical transparency, the platform incorporates interpretability methods:
- **AFIB Diagnosis:** Attention weights concentrated heavily (**~68%**) in leads **II, V1, and aVF**, matching clinical reliance on irregular R-R intervals and absent P-waves.
- **LBBB Diagnosis:** Attention focused on broad, notched QRS morphologies across lateral leads **V5, V6, and I**.
- **Error Analysis:** Low signal quality records exhibited dispersed temporal attention and distinctly lower confidence scores, allowing the Safe-Fail engine to intervene.

---

## ⚡ Hardware & Performance Benchmarks

- **Development Rig:** Intel Core i7-14700HX CPU (28 threads), NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM), 32 GB RAM.
- **Operating System:** Windows 11 (Python 3.12, PyTorch CUDA).
- **VRAM Utilization:** 1D convolutions maintain a lean **3–5 GB VRAM** profile (runs comfortably on 8 GB cards).
- **Training Time:**
  - *Pre-training (~37,000 records):* ~2 hours (30 Epochs, AMP enabled)
  - *Fine-Tuning (5,000 records):* ~20 minutes (20–30 Epochs)
  - *HDF5 Caching Impact:* Reduced epoch training duration from **12 minutes to under 2 minutes**.
- **Inference Latency:**
  - GPU: **5 – 10 ms**
  - Modern CPU: **~50 ms** *(Completely viable for real-time bedside and portable systems)*

---

## 📂 Project Directory Structure

```bash
├── augment.py             # Data augmentation routines (Noise, Baseline Wander, Scale, Shift)
├── dataset.py             # HDF5-backed high-throughput PyTorch Dataset
├── decision.py            # Temperature scaling, threshold sweep & Safe-Fail rejection logic
├── evaluate.py            # Metric evaluation suite (Macro-F1, ROC-AUC, MCC, Confusion Matrix)
├── explainability.py      # Attention Rollout & model interpretability module
├── genelsema.txt          # Architectural design specifications & schema notes
├── hdf5_builder.py        # WFDB / PhysioNet raw data ingestion and HDF5 generator
├── loss.py                # Class-weighted Asymmetric Loss (ASL) module
├── model.py               # Hybrid ResNet-1D + CBAM + Transformer architecture
├── preprocess.py          # Bandpass (0.5-45Hz), Notch (50Hz), Z-score filtering
├── rapor_kaynaklar.txt    # Compute metrics, hardware details & PDR report notes
├── train.py               # Pretraining and fine-tuning execution pipeline
├── README.md              # Turkish documentation
└── README_EN.md           # English documentation
```

---

## 🚀 Installation & Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/Devre181/EKG-Tarayici.git
cd EKG-Tarayici

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install wfdb h5py scikit-learn scipy numpy pandas
```

### 2. Preprocess & Cache Data (HDF5)
```bash
python hdf5_builder.py
```

### 3. Model Training
```bash
python train.py
```

---

## 📚 References & Literature

1. **Ribeiro, A. H., et al. (2020).** Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications*, 11(1), 1760.
2. **Strodthoff, N., et al. (2021).** Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. *IEEE JBHI*, 25(5), 1519–1528.
3. **Zhu, H., et al. (2020).** Automatic multilabel electrocardiogram diagnosis of heart rhythm or conduction abnormalities with deep learning. *The Lancet Digital Health*, 2(7), e348–e357.
4. **Zhou, F., & Fang, D. (2025).** Classification of multi-lead ECG based on multiple scales and hierarchical feature convolutional neural networks. *Scientific Reports*, 15, 16418.
5. **Najia, M., & Faouzi, B. (2025).** An Enhanced Hybrid Model Combining CNN, BiLSTM, and Attention Mechanism for ECG Segment Classification. *Biomedical Engineering and Computational Biology*, 16, 1–14.
6. **Alghieth, M. (2025).** DeepECG-Net: A Hybrid Transformer-Based Deep Learning Model for Real-Time ECG Anomaly Detection. *Scientific Reports*, 15(1), 20714.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Open for academic and clinical research.

<div align="center">
  <b>TEKNOFEST 2026 — Team Devre181 🚀</b>
</div>
