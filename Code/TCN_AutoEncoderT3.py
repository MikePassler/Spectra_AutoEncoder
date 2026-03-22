"""
TCN_AutoEncoderT3.py

Script version of the Jupyter notebook `TCN_AutoEncoderT3.ipynb`.

This file implements a Temporal Convolutional Network (TCN) autoencoder
for processing Cassini CDA mass spectra. It mirrors the notebook flow:
- configuration
- data loading (Parquet / dummy fallback)
- preprocessing
- model definition
- training
- specialized 'type-3' experiment and evaluation

Run: python3 /Users/mike/AutoEncoder/TCN_AutoEncoderT3.py

Note: This script tries to import optional dependencies and will fall back
or raise clear errors if critical packages are missing.
"""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist


# ==========================================
# CONFIGURATION
# ==========================================
Input_Length = 1000
Batch_Size = 128
Learning_Rate = 1e-3
Epochs = 20

# TCN Specific Architecture Choices (Adjustable)
Latent_Dim = 32
Channel_List = [32, 64, 64, 128, 256, 256]
Kernel_Size = 5
Dropout_Rate = 0.1
Slope = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------
# Data Loading
# -------------------------
file_path = 'cda_qm_spectra_pre2008277_train_lvl2.parquet'

if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    print(f"Loaded data shape: {df.shape}")

    if 'spectrum' in df.columns:
        # ensure a stackable array
        spectra_raw = np.stack(df['spectrum'].values)
    else:
        raise ValueError(f"Column 'spectrum' not found. Available columns: {df.columns}")
else:
    print(f"Warning: File {file_path} not found. Creating dummy data for demonstration.")
    df = pd.DataFrame({
        'class': np.random.choice(['Type A', 'Type B', 'Type C'], size=1000),
        'spectrum': [np.random.rand(Input_Length).astype(np.float32) for _ in range(1000)]
    })
    spectra_raw = np.stack(df['spectrum'].values)
    print("Created dummy data.")

if 'class' in df.columns:
    print("\nClass distribution:")
    print(df['class'].value_counts())
else:
    print("Column 'class' not found in dataframe.")


# -------------------------
# Preprocessing
# -------------------------

def preprocess_spectra(spectra, target_length=1000):
    processed = []

    for spec in spectra:
        if len(spec) >= target_length:
            s = spec[:target_length]
        else:
            s = np.pad(spec, (0, target_length - len(spec)), 'constant')

        # Denoise
        window_length = min(11, len(s) if len(s) % 2 == 1 else len(s) - 1)
        if window_length >= 5:
            s = savgol_filter(s, window_length=window_length, polyorder=3)

        s = np.log1p(np.maximum(s, 0))

        max_val = np.max(s)
        if max_val > 0:
            s = s / max_val

        processed.append(s)

    return np.array(processed, dtype=np.float32)

print("Preprocessing data...")
X_train = preprocess_spectra(spectra_raw, target_length=Input_Length)
print(f"Processed Data Shape: {X_train.shape}")

# Class weighting
print("Calculating class weights (Balanced)...")
sample_weights_tensor = None

if 'class' in df.columns:
    y_train_classes = df['class'].values
    unique_classes = np.unique(y_train_classes)
    weights_per_class = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train_classes)
    class_weights = dict(zip(unique_classes, weights_per_class))

    weights_np = df['class'].map(class_weights).values.astype(np.float32)
    weights_np = weights_np / weights_np.mean()
    sample_weights_tensor = torch.from_numpy(weights_np)

    print(f"  Class weights generated. (Max: {weights_np.max():.2f}, Min: {weights_np.min():.2f})")
else:
    print("  'class' column missing. Using uniform weights.")
    sample_weights_tensor = torch.ones(len(X_train), dtype=torch.float32)

# DataLoader
dataset = TensorDataset(torch.from_numpy(X_train).unsqueeze(1), sample_weights_tensor)
loader = DataLoader(dataset, batch_size=Batch_Size, shuffle=True)
print("DataLoader created.")


# -------------------------
# Model Definition
# -------------------------
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.LeakyReLU(Slope)
        self.dropout1 = nn.Dropout(dropout)

        pad2 = (kernel_size - 1) * dilation // 2
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=1, padding=pad2, dilation=dilation)
        self.relu2 = nn.LeakyReLU(Slope)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.stride = stride

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        if self.stride > 1 and self.downsample is None:
            res = res[:, :, ::self.stride]
        elif self.stride > 1 and self.downsample is not None:
            res = F.interpolate(res, size=out.shape[2], mode='linear', align_corners=False)

        return self.relu2(out + res)


class TCNAutoEncoder(nn.Module):
    def __init__(self, input_len, latent_dim, channels, kernel_size):
        super(TCNAutoEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        current_channels = 1

        for ch in channels:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(current_channels, ch, kernel_size, stride=2, padding=kernel_size//2),
                    nn.BatchNorm1d(ch),
                    nn.LeakyReLU(Slope),
                    nn.Dropout(Dropout_Rate),
                )
            )
            current_channels = ch

        # compute flatten dim
        self._calculate_flatten_dim(input_len)
        print(f"Flattened dimension before bottleneck: {self.flatten_dim} ({current_channels} x {self.final_len})")

        self.bottleneck = nn.Linear(self.flatten_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_reshape_ch = current_channels
        self.decoder_reshape_len = self.final_len

        self.decoder_layers = nn.ModuleList()
        reversed_channels = list(reversed(channels))

        for i in range(len(reversed_channels)):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i+1] if i < len(reversed_channels)-1 else 1

            is_last = (i == len(reversed_channels)-1)
            activation = nn.Sigmoid() if is_last else nn.LeakyReLU(Slope)

            self.decoder_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_ch),
                    activation
                )
            )

    def _calculate_flatten_dim(self, input_len):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_len)
            out = dummy_input
            for layer in self.encoder_layers:
                out = layer(out)
            self.final_len = out.shape[2]
            self.flatten_dim = out.numel()

    def forward(self, x):
        out = x
        for layer in self.encoder_layers:
            out = layer(out)

        out = out.view(out.size(0), -1)
        latent = self.bottleneck(out)

        out = self.decoder_input(latent)
        out = out.view(out.size(0), self.decoder_reshape_ch, self.decoder_reshape_len)

        for layer in self.decoder_layers:
            out = layer(out)

        if out.size(2) != x.size(2):
            out = F.interpolate(out, size=x.size(2), mode='linear', align_corners=False)

        return out, latent


print("=" * 70)
print("TCN AutoEncoder Architecture")
print("=" * 70)

model = TCNAutoEncoder(input_len=Input_Length, latent_dim=Latent_Dim, channels=Channel_List, kernel_size=Kernel_Size).to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal Parameters: {total_params:,}")
print(f"Compression: {Input_Length} floats -> {Latent_Dim} floats (Ratio: {Input_Length/Latent_Dim:.1f}x)")
print("=" * 70)


# -------------------------
# Training
# -------------------------
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)
criterion = nn.MSELoss(reduction='none')

print("=" * 70)
print(f"Training TCN Autoencoder (Latent Dim: {Latent_Dim})")
print(f"Epochs: {Epochs}")
print("=" * 70)

for epoch in range(Epochs):
    model.train()
    total_loss = 0.0
    for batch in loader:
        img = batch[0].to(device)
        if len(batch) > 1:
            weights = batch[1].to(device)
        else:
            weights = torch.ones(img.size(0), device=device)

        optimizer.zero_grad()
        recon, latent = model(img)
        loss_per_sample = criterion(recon, img).mean(dim=[1, 2])
        weighted_loss = (loss_per_sample * weights).mean()

        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{Epochs}, Loss: {avg_loss:.6f}")

print("\n✓ Training complete!")


# -------------------------
# Evaluation & Specialized type-3 experiment
# -------------------------
print('\nStarting specialized type-3 experiment...')

# Prepare label encoder
if 'class' in df.columns:
    le = LabelEncoder().fit(df['class'].astype(str).values)
    print('Fitted LabelEncoder on df.class with', len(le.classes_), 'classes')
else:
    le = LabelEncoder()
    print('Created empty LabelEncoder (will fit later if needed)')

# Detect type-3 labels
unique_classes = np.unique(df['class'].astype(str)) if 'class' in df.columns else []
three_candidates = [c for c in unique_classes if '3' in str(c)]
subtype_candidates = [c for c in three_candidates if str(c).startswith('3-')]
generic_candidates = [c for c in three_candidates if c not in subtype_candidates]

print(f"Found classes containing '3': {three_candidates}")
print(f"Detected subtype labels (start with '3-'): {subtype_candidates}")
print(f"Detected generic candidates: {generic_candidates}")

if len(subtype_candidates) == 0:
    subtype_candidates = [c for c in unique_classes if re.search(r'3[-_]', str(c))]
    generic_candidates = [c for c in three_candidates if c not in subtype_candidates]
    print(f"Fallback subtype candidates: {subtype_candidates}")

if len(three_candidates) == 0:
    raise RuntimeError("No classes containing the character '3' were found in df['class']. Please check your labels.")

generic_label = '3'
if generic_label not in df['class'].astype(str).values:
    print("Warning: hardcoded generic label '3' not found. Falling back to heuristic.")
    if len(generic_candidates) > 0:
        counts = df['class'].value_counts()
        generic_label = sorted(generic_candidates, key=lambda c: -counts.get(c, 0))[0]
        print(f"Falling back to detected generic label: {generic_label}")
    else:
        generic_label = None
        print("No suitable generic candidate found; proceeding without excluding any label.")
else:
    print(f"Using hardcoded generic label: {generic_label}")

is_three = df['class'].astype(str).str.contains(r"\b3\b|^3|3[-_]", regex=True)
is_subtype = df['class'].astype(str).apply(lambda x: any([str(x).startswith(s) for s in subtype_candidates]))
if generic_label is not None:
    is_generic = df['class'].astype(str) == str(generic_label)
else:
    is_generic = (is_three & (~is_subtype))

num_subtypes_in_data = is_subtype.sum()
print(f"Subtype samples found in dataset: {num_subtypes_in_data}")

train_mask = ~is_generic
test_mask = is_three
print(f"Train size (excluding generic type-3): {train_mask.sum()} samples")
print(f"Test size (type-3 samples only): {test_mask.sum()} samples")

X_all = X_train.astype(np.float32)
if 'weights_np' in globals():
    weights_all = weights_np.astype(np.float32)
elif 'sample_weights_tensor' in globals():
    weights_all = sample_weights_tensor.numpy().astype(np.float32)
else:
    weights_all = np.ones(len(X_all), dtype=np.float32)

train_X = X_all[train_mask]
train_w = weights_all[train_mask]

test_X = X_all[test_mask]
test_labels = df['class'].astype(str).values[test_mask]

train_dataset = TensorDataset(torch.from_numpy(train_X).unsqueeze(1), torch.from_numpy(train_w))
train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)

# Retrain model on training split (excluding generic)
print('\nReinitializing model and training on train split (no generic type-3)')
model = TCNAutoEncoder(input_len=Input_Length, latent_dim=Latent_Dim, channels=Channel_List, kernel_size=Kernel_Size).to(device)
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)
criterion = nn.MSELoss(reduction='none')

for epoch in range(Epochs):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        imgs = batch[0].to(device)
        ws = batch[1].to(device)
        optimizer.zero_grad()
        recon, _ = model(imgs)
        loss_per_sample = criterion(recon, imgs).mean(dim=[1, 2])
        loss = (loss_per_sample * ws).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{Epochs}  Train Loss: {epoch_loss/len(train_loader):.6f}")

print('\nTraining on split complete.')

# Encode training set to compute latents
model.eval()
with torch.no_grad():
    train_enc_ds = TensorDataset(torch.from_numpy(train_X).unsqueeze(1))
    train_enc_loader = DataLoader(train_enc_ds, batch_size=Batch_Size, shuffle=False)
    train_latents = []
    for b in train_enc_loader:
        imgs = b[0].to(device)
        _, lat = model(imgs)
        train_latents.append(lat.cpu().numpy())
    train_latent_all = np.concatenate(train_latents, axis=0)

train_labels = df['class'].astype(str).values[train_mask]

subtype_list = sorted([s for s in subtype_candidates if s in df['class'].values])
print(f"Subtype list used for centroid computation: {subtype_list}")

if len(subtype_list) == 0:
    print("No subtype labels found in training set — cannot compute subtype centroids. Exiting special evaluation.")
else:
    subtype_centroids = {}
    for sub in subtype_list:
        mask_sub = train_labels == sub
        if mask_sub.sum() == 0:
            print(f"Warning: Subtype {sub} has no training samples; skipping")
            continue
        centroid = train_latent_all[mask_sub].mean(axis=0)
        subtype_centroids[sub] = centroid
    print(f"Computed {len(subtype_centroids)} subtype centroids.")

    # Encode test samples
    test_ds = TensorDataset(torch.from_numpy(test_X).unsqueeze(1))
    test_loader = DataLoader(test_ds, batch_size=Batch_Size, shuffle=False)
    test_latents = []
    with torch.no_grad():
        for b in test_loader:
            imgs = b[0].to(device)
            _, lat = model(imgs)
            test_latents.append(lat.cpu().numpy())
    test_latent_all = np.concatenate(test_latents, axis=0)

    scaler_lat = StandardScaler().fit(train_latent_all)
    train_lat_scaled = scaler_lat.transform(train_latent_all)
    test_lat_scaled = scaler_lat.transform(test_latent_all)

    centroid_names = list(subtype_centroids.keys())
    centroid_matrix = np.vstack([subtype_centroids[n] for n in centroid_names])
    centroid_matrix_scaled = scaler_lat.transform(centroid_matrix)

    dists = cdist(test_lat_scaled, centroid_matrix_scaled, metric='euclidean')
    assign_idx = np.argmin(dists, axis=1)
    assigned_subtypes = [centroid_names[i] for i in assign_idx]

    test_labels_arr = np.array(test_labels)
    is_test_subtype = np.array([lbl in centroid_names for lbl in test_labels_arr])
    subtype_test_mask = is_test_subtype
    generic_test_mask = ~is_test_subtype

    print('\nAssignment summary for type-3 test set:')
    cnt_assigned = Counter(assigned_subtypes)
    for k, v in cnt_assigned.items():
        print(f"  Assigned to {k}: {v} samples")

    if subtype_test_mask.sum() > 0:
        true_subtype_labels = test_labels_arr[subtype_test_mask]
        pred_for_true = np.array(assigned_subtypes)[subtype_test_mask]
        acc = (pred_for_true == true_subtype_labels).mean()
        print(f"\nAssignment accuracy among true subtype samples: {acc:.4f} (N={len(true_subtype_labels)})")

    if generic_test_mask.sum() > 0:
        gen_assigned = np.array(assigned_subtypes)[generic_test_mask]
        print('\nGeneric samples assigned counts:')
        for k, v in Counter(gen_assigned).items():
            print(f"  {k}: {v}")

    if subtype_test_mask.sum() > 1:
        try:
            sil = silhouette_score(test_lat_scaled[subtype_test_mask], le.transform(test_labels_arr[subtype_test_mask]))
            print(f"\nSilhouette among subtype test samples: {sil:.4f}")
        except Exception as e:
            print('Silhouette score failed:', e)

    # Visualizations (t-SNE may take time)
    print('\nRendering t-SNE plots (may take a moment)...')
    tsne_input = test_lat_scaled if test_lat_scaled.shape[0] <= 10000 else test_lat_scaled[np.random.choice(test_lat_scaled.shape[0], 10000, replace=False)]
    labels_for_tsne = test_labels_arr if test_lat_scaled.shape[0] <= 10000 else test_labels_arr[np.random.choice(test_lat_scaled.shape[0], 10000, replace=False)]
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(tsne_input)

    plt.figure(figsize=(10, 6))
    unique_labels_tsne = np.unique(labels_for_tsne)
    for lbl in unique_labels_tsne:
        mask_lbl = labels_for_tsne == lbl
        plt.scatter(tsne_2d[mask_lbl, 0], tsne_2d[mask_lbl, 1], label=lbl, s=15, alpha=0.7)
    plt.title('t-SNE of Type-3 test samples (true labels)')
    plt.legend(fontsize=8)
    plt.show()

    plt.figure(figsize=(10, 6))
    assigned_for_plot = np.array(assigned_subtypes) if test_lat_scaled.shape[0] <= 10000 else np.array(assigned_subtypes)[np.random.choice(len(assigned_subtypes), 10000, replace=False)]
    tsne_plot = tsne_2d
    for lbl in np.unique(assigned_for_plot):
        mask_lbl = assigned_for_plot == lbl
        plt.scatter(tsne_plot[mask_lbl, 0], tsne_plot[mask_lbl, 1], label=lbl, s=15, alpha=0.7)
    plt.title('t-SNE of Type-3 test samples (assigned subtypes)')
    plt.legend(fontsize=8)
    plt.show()

    if subtype_test_mask.sum() > 0:
        true_labels_sub = test_labels_arr[subtype_test_mask]
        pred_labels_sub = np.array(assigned_subtypes)[subtype_test_mask]
        cm = confusion_matrix(true_labels_sub, pred_labels_sub, labels=centroid_names)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, xticklabels=centroid_names, yticklabels=centroid_names, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted (centroid)')
        plt.ylabel('True subtype')
        plt.title('Confusion matrix for subtype assignments (test)')
        plt.show()

    if generic_test_mask.sum() > 0:
        gen_counts = Counter(gen_assigned)
        labels = list(gen_counts.keys())
        vals = [gen_counts[l] for l in labels]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, vals, color='tab:orange')
        plt.title('Assigned subtypes for GENERIC type-3 samples')
        plt.ylabel('Count')
        plt.xlabel('Assigned subtype')
        plt.show()

print('\nSpecialized type-3 experiment complete.')
