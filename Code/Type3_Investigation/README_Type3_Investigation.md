# TCN AutoEncoder for Cassini CDA Mass Spectra

This repository contains `TCN_AutoEncoderT3.py`, a script designed to process, encode, and classify mass spectra from the Cassini Cosmic Dust Analyzer (CDA) using a Temporal Convolutional Network (TCN) Autoencoder.

## Overview

The `TCN_AutoEncoderT3.py` script performs the following core tasks:
1. **Data Loading & Preprocessing**: Loads spectral data from a Parquet file, pads/truncates spectra to a fixed length of 1000, denoises using a Savitzky-Golay filter, and applies `log1p` and max-scaling normalization.
2. **Autoencoder Training**: Trains a 1D Convolutional Autoencoder (TCN) to compress the 1000-dimensional spectra into a 32-dimensional latent space.
3. **Specialized Type-3 Experiment**: 
   - **Purpose**: The overarching purpose of this investigation is to see if generic "Type 3" spectra can be accurately assigned into more specific Type-3 subtypes (e.g., 3-1, 3-A).
   - Isolates "Type-3" spectral classes from the dataset.
   - Specifically evaluates how well the latent space separates generic "3" labels from "3-subtypes" (e.g., 3-1, 3-A).
   - Uses a Logistic Regression classifier trained on the latent embeddings to classify the subtypes.
4. **Visualization**: Generates t-SNE projections and confusion matrices, saving them automatically.

## Prerequisites

Ensure you have the following dependencies installed in your Python environment:

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn pyarrow
```
*(Note: `pyarrow` or `fastparquet` is required for pandas to read `.parquet` files).*

## Required Files

- `TCN_AutoEncoderT3.py`: The main execution script.
- `cda_qm_spectra_pre2008277_train_lvl2.parquet`: The dataset. It should be placed in the same directory as the script. *(If not found, the script will generate heavily-randomized dummy data for demonstration purposes but results will not be meaningful).*

## How to Run

### 1. Standard Execution (Local or Interactive Node)
You can run the script directly using Python. Make sure you are in the directory containing the script and the data file.

```bash
cd /home/passlem01/Scripts
python3 TCN_AutoEncoderT3.py
```

### 2. SLURM Cluster Execution
Since this script is optimized for multi-threading (`Num_Workers = 8`) and can utilize GPUs if available, it is recommended to run it via a SLURM batch script on a cluster.

```bash
sbatch run_type3_validation.sbatch
```

## Outputs

The script operates in a headless mode (no pop-up windows) and saves all generated artifacts to an `./outputs/` directory.

Once the script completes, check the `outputs/` folder for the following files:
* `tsne_type3_true_labels.png`: A t-SNE projection colored by the actual dataset labels.
* `tsne_type3_assigned_subtypes.png`: A t-SNE projection colored by the Logistic Regression predictions.
* `confusion_matrix_subtypes.png`: A heatmap showing the classification accuracy between the known subtypes.
* `generic_type3_assignments.png`: A bar chart showing how the generic "3" class was distributed among the specific subtypes by the classifier.
