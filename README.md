# Autonomous Spectral Interpretation: Unsupervised Machine Learning for Cassini CDA & Europa Clipper SUDA

## 🔬 Mission Context & Scientific Objectives

This repository contains state-of-the-art unsupervised machine learning pipelines and autoencoder architectures designed for the autonomous analysis of high-velocity ice grain impact mass spectra. The project processes data from the **Cassini Cosmic Dust Analyzer (CDA)** and serves as an analogue framework for the upcoming **Europa Clipper Surface Dust Analyzer (SUDA)**.

### Target Environment
- **Enceladus Plume & Europa Analogues**: Identifying bio-essential elements and macroscopic organic molecules in subsurface ocean materials.
- **Constraints**: Evaluates **Cation mode only**, handling high instrumental background noise and complex matrix interferences from salt clusters in entirely unlabeled datasets.

### Chemical Fingerprints & Latent Space Interpretation
The autoencoder latent space is rigorously designed to capture and distinguish distinct chemical fingerprints embedded within the spectra:
- **Inorganic Matrix**: Water standard clusters (m/z 19, 37, 55), salinity markers (Na+ 23, K+ 39), bio-essential phosphorus ([Na3PO4]+ at m/z 165), and Iron Redox Tracers (Fe(III) / Fe(II)).
- **Organic & Biosignatures**: High Mass Organic Cations (>200 u with ~12.5 u spacing), aromatic parents vs. derivatives, and low-mass volatile ratios. Trace amino acids (e.g., Histidine, Arginine) appearing as sodiated molecular peaks.

### Unsupervised Clustering Strategy
With significant instrumental noise and matrix interferences, this project utilizes autoencoders to compress the 1000-dimensional "spectra" vector to isolate chemical signals. Anomaly detection via reconstruction error flags rare grains vs. common water/salt grains. The clustering goals target four main grain types:
- **Cluster A (Pure Water-Ice)**: Standard background grains.
- **Cluster B (Salt-Rich)**: Elevated Na/K intensities.
- **Cluster C (HMOC-Bearing)**: Refractory organic films.
- **Cluster D (Biosignature-Rich)**: Potential biotic indicators.

---

## 🚀 Repository Capabilities & Model Architectures

Due to the lack of labels and complex spectral overlaps (such as the m/z 219 interference between sodiated Arginine and salt clusters), this project employs multiple autoencoder architectures. These models must implicitly learn interpretative rules like the **Mass Deficit Rule** (differentiating positive offset organics from negative offset abiotic salt clusters) and **Peak Asymmetry** at 600-800 m/Δm resolution boundaries.

### 🧠 Implemented Architectures

1. **Transformer Autoencoder** (`Code/Transformer_Autoencoder.ipynb`)
   - **Mechanism**: Self-attention mechanisms with positional encodings.
   - **Advantage**: Superior peak preservation and global context modeling. Ideal for detecting widely separated isotopic patterns and trace organic fragments against high-salt backgrounds.

2. **VQ-VAE (Vector Quantized Variational Autoencoder)** (`Code/VQ-VAE.ipynb`)
   - **Mechanism**: Projects continuous spectra into a discrete learned codebook (512 entries).
   - **Advantage**: Provides the **best cluster separation** (highest Silhouette Score). Ensures discrete classification of pure water vs. organic-rich grains.

3. **Temporal Convolutional Network (TCN) Autoencoder** (`Code/TCN_AutoEncoder.ipynb` & `Code/TCN_AutoEncoderT3.py`)
   - **Mechanism**: Progressive 1D convolutional downsampling capturing temporal/sequential mass spacing. Tested across multiple latent dimensions (128, 64, 32).
   - **Advantage**: Highly efficient compression (up to 31.3x) while maintaining strong nearest-neighbor similarity and high fidelity reconstruction.

4. **Deep Embedded Autoencoder** (`Code/Deep_embeddedauto (1).ipynb`)
   - **Mechanism**: CNN bottleneck supplemented with an optional Student's t-distribution clustering layer.
   - **Advantage**: Achieves the **best feature discriminability** (Linear Probe F1) and lowest baseline reconstruction error.

5. **Denoising Autoencoder (DAE)** (`Code/Denoising Autoencoder.ipynb`)
   - **Mechanism**: Injects Gaussian noise during training.
   - **Advantage**: Crucial for stripping uninformative instrumental artifacts and electrical noise inherent in CDA data.

---

## 📁 Repository Structure

```text
├── Code/                             # Core Jupyter notebooks and python scripts
│   ├── Deep_embeddedauto (1).ipynb   # Deep Embedded clustering model
│   ├── Denoising Autoencoder.ipynb   # DAE for noise-robust features
│   ├── Transformer_Autoencoder.ipynb # Self-attention models
│   ├── VQ-VAE.ipynb                  # Discrete latent representation
│   ├── Multi_Model_TCN...ipynb       # TCN compression comparison
│   ├── TCN_AutoEncoderT3.py          # Script logic for subtype classification
│   └── Type3_Investigation/          # Granular analysis of "Type-3" spectral classes
├── Dataset/                          # Parquet data directory
│   └── cda_qm_spectra_pre2008277_train_lvl2.parquet
└── ReadME files/                     # Historical architecture-specific documentations
```

## ⚙️ Quick Start

**Prerequisites:**
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn scipy pyarrow
```

**Workflow:**
1. Place the required `.parquet` dataset in the `Dataset/` folder.
2. The dataset undergoes pre-processing (Savitzky-Golay filtering, log1p transformation, and max-scaling normalizations).
3. Execute any of the `.ipynb` notebooks in `Code/` to train models, evaluate latent representation discriminability, and visualize projections via t-SNE.

*Note: Pre-trained model checkpoints (`.pth` files) are tracked separately outside of version control due to file size constraints. Contact the repository owner for access to the trained weights (~400MB total).*
