# Unsupervised Machine Learning for Cassini CDA Mass Spectra Analysis

This repository contains implementations of various autoencoder architectures for analyzing mass spectra from the **Cassini Cosmic Dust Analyzer (CDA)**. The project explores different unsupervised learning approaches to extract meaningful features from high-dimensional, sparse, and noisy spectral data.

## 🎯 Project Overview

The Cassini spacecraft's Cosmic Dust Analyzer collected mass spectra from dust grains in Saturn's system. This project applies state-of-the-art autoencoder architectures to:

- Learn compact, noise-robust latent representations of mass spectra
- Extract chemical signatures and spectral patterns
- Cluster and classify spectra based on learned features
- Evaluate reconstruction quality and feature discriminability

## 📁 Directory Structure

```
ML_Unsupervised/
├── Code/
│   ├── Deep_embeddedauto (1).ipynb       # Deep Embedded Autoencoder
│   ├── Denoising Autoencoder.ipynb       # Denoising Autoencoder (DAE)
│   ├── Transformer_Autoencoder.ipynb     # Transformer-based Autoencoder
│   └── VQ-VAE.ipynb                      # Vector Quantized VAE
├── Dataset/
│   └── cda_qm_spectra_pre2008277_train_lvl2.parquet  # Training data
├── Models/
│   ├── deep_embedded_autoencoder_latent128.pth
│   ├── denoising_autoencoder_latent128_noise0.3_weighted.pth
│   ├── transformer_autoencoder_d256_h8_latent128.pth
│   └── vqvae_embed128_codebook512.pth
└── README.md
```
(The model files are too big to be uploaded on GitHub, please let me know if they are needed)
## 🧠 Implemented Architectures

### 1. Deep Embedded Autoencoder
**File:** [Code/Deep_embeddedauto (1).ipynb](Code/Deep_embeddedauto%20(1).ipynb)

A CNN-based autoencoder with a compact bottleneck layer designed for clustering and dimensionality reduction.

**Key Features:**
- CNN encoder for local peak pattern recognition
- 128-dimensional bottleneck for optimal clustering
- CNN decoder for reconstruction
- Optional clustering layer with Student's t-distribution
- Designed to overcome the curse of dimensionality

**Model:** `Models/deep_embedded_autoencoder_latent128.pth`

---

### 2. Denoising Autoencoder (DAE)
**File:** [Code/Denoising Autoencoder.ipynb](Code/Denoising%20Autoencoder.ipynb)

Trains on corrupted inputs to learn noise-robust features by reconstructing clean signals.

**Key Features:**
- Gaussian noise injection during training
- CNN encoder/decoder architecture
- 128-dimensional latent space
- MSE reconstruction loss
- Robust feature learning and regularization

**Advantages:**
- Noise-resistant representations
- Better generalization
- Prevents overfitting through noise regularization

**Model:** `Models/denoising_autoencoder_latent128_noise0.3_weighted.pth`

---

### 3. Transformer Autoencoder
**File:** [Code/Transformer_Autoencoder.ipynb](Code/Transformer_Autoencoder.ipynb)

Leverages self-attention mechanisms for global context and better peak preservation.

**Key Features:**
- Embedding layer for spectrum projection (d_model = 128 or 256)
- Positional encodings for m/z position awareness
- 4-6 transformer encoder layers with 8 attention heads
- Global average pooling bottleneck
- Transformer decoder with upsampling
- Intensity-weighted MSE + L1 sparsity loss

**Advantages:**
- Global attention across entire spectrum
- Better preservation of sparse chemical peaks
- Long-range dependency modeling (isotope patterns)
- Improved SNR for chemical classes

**Model:** `Models/transformer_autoencoder_d256_h8_latent128.pth`

---

### 4. VQ-VAE (Vector Quantized Variational Autoencoder)
**File:** [Code/VQ-VAE.ipynb](Code/VQ-VAE.ipynb)

Uses discrete latent codes from a learned codebook for better interpretability and compression.

**Key Features:**
- CNN encoder for continuous feature extraction
- Vector quantization layer with learned codebook
- Straight-through estimator for gradient flow
- CNN decoder for reconstruction
- Triple loss: reconstruction + codebook + commitment

**Advantages:**
- Discrete latent space (512 codebook entries)
- Better interpretability of learned codes
- Compact vocabulary of spectral features
- Avoids posterior collapse problem of standard VAEs

**Model:** `Models/vqvae_embed128_codebook512.pth`

## 📊 Dataset

**File:** `Dataset/cda_qm_spectra_pre2008277_train_lvl2.parquet`

Contains pre-processed mass spectra from Cassini CDA observations before day 277 of 2008.

**Data Format:** Parquet file with spectral measurements (1000 bins per spectrum)

## 🔄 Common Pipeline

All notebooks follow a similar workflow:

1. **Data Loading:** Read Parquet files containing mass spectra
2. **Preprocessing:**
   - Savitzky-Golay filtering for noise reduction
   - Log transformation for dynamic range compression
   - Normalization for consistent scaling
3. **Model Architecture:** Build and configure the autoencoder
4. **Training:** Optimize model parameters with appropriate loss functions
5. **Evaluation:**
   - Silhouette score for cluster quality
   - Linear probe for feature discriminability
   - Reconstruction error (MSE/MAE)
   - Signal-to-noise ratio (SNR) analysis

## 🏃 Getting Started

### Prerequisites

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy
pip install pyarrow  # For parquet support
```

### Running the Notebooks

1. Open any notebook in Jupyter or VS Code
2. Ensure the dataset path points to `../Dataset/cda_qm_spectra_pre2008277_train_lvl2.parquet`
3. Run cells sequentially to:
   - Load and preprocess data
   - Build the model architecture
   - Train the autoencoder
   - Evaluate performance metrics
   - Visualize results

## 📈 Model Comparison

| Model | Latent Dim | Trainable Parameters | Special Features | Best For |
|-------|-----------|---------------------|------------------|----------|
| Deep Embedded | 128 | 8,377,857 | Clustering layer | Dimensionality reduction, clustering |
| Denoising AE | 128 | 4,215,809 | Noise injection | Robust feature learning, noise resistance |
| Transformer | 128 | 1,491,841 | Self-attention | Peak preservation, global context |
| VQ-VAE | 128 | 169,217 | Discrete codebook | Interpretability, compression |

## 🎯 Evaluation Metrics

All models are evaluated using:

- **Silhouette Score:** Measures cluster separation quality (-1 to 1, higher is better)
- **Linear Probe Accuracy/F1:** Tests feature discriminability with a linear classifier
- **Reconstruction Error:** MSE/MAE between input and reconstructed spectra

## 📊 Performance Results

### Silhouette Score (Higher is Better)

| Rank | Model | Score |
|------|-------|-------|
| 1st | VQ-VAE | 0.0403 |
| 2nd | Transformer | 0.0206 |
| 3rd | Deep Embedded | -0.0277 |
| 4th | Denoising | -0.0725 |

### Linear Probe F1 Score (Higher is Better)

| Rank | Model | Score |
|------|-------|-------|
| 1st | Deep Embedded | 0.8860 |
| 2nd | VQ-VAE | 0.8672 |
| 3rd | Transformer | 0.8625 |
| 4th | Denoising | 0.8525 |

### Reconstruction Error (Lower is Better)

| Rank | Model | Error |
|------|-------|-------|
| 1st | Deep Embedded | 0.00272 |
| 2nd | VQ-VAE | 0.007175 |
| 3rd | Transformer | 0.021185 |
| 4th | Denoising | 0.069 |

### Key Insights

- **VQ-VAE** achieves the best cluster separation (Silhouette Score), indicating its discrete latent codes effectively group similar spectra
- **Deep Embedded Autoencoder** excels at feature discriminability (F1 Score) and reconstruction quality, making it ideal for classification tasks
- **Transformer** provides balanced performance across all metrics with superior peak preservation
- **Denoising Autoencoder** sacrifices reconstruction fidelity for noise-robust features, as expected from its design


## 📝 Notes

- All models use a latent dimension of 128 
- Training data is from pre-2008 day 277 Cassini CDA observations
- Models are saved in PyTorch format (`.pth` files)
- Preprocessing includes Savitzky-Golay filtering, log transformation, and normalization

## 🔬 Scientific Context

The Cassini spacecraft studied Saturn and its moons from 2004-2017. The Cosmic Dust Analyzer (CDA) measured the composition of dust grains by ionizing them and measuring mass spectra. This unsupervised learning approach helps identify chemical signatures and classify different types of dust particles without labeled training data.

## 📄 License

This project is for research and educational purposes.

---

**Last Updated:** January 2026
