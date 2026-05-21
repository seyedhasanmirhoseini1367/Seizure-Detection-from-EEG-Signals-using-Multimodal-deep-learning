# Seizure Detection from EEG Signals using Multimodal Deep Learning

Seizure detection from EEG signals is a vital task in the diagnosis and management of epilepsy.
This project presents a multimodal deep learning framework that classifies **Seizure** vs
**Lateralized Periodic Discharge (LPD)** patterns from 19-channel referential EEG recordings.

Three transformer-based architectures are developed and compared:

| Model | Modality | Accuracy |
|---|---|---|
| LSTM → Transformer | Raw temporal signal | **79.15 %** |
| CNN → Transformer | STFT spectrogram | **90.12 %** |
| Fusion (LSTM + CNN → Self-Attention) | Both | **82.51 %** |

---

## Architecture

### Model 1 — LSTM → Transformer

Each EEG channel's sliding-window time series is encoded by a shared LSTM.
The per-channel last-step hidden states are stacked into a sequence and fed to a
Transformer encoder that models inter-channel temporal relationships.

<img width="619" height="1007" alt="LSTM-Transformer architecture" src="https://github.com/user-attachments/assets/622ffde5-90cb-4757-a507-91d3b1e5b902" />

### Model 2 — CNN → Transformer

Each EEG channel's STFT spectrogram is processed independently by a two-stage CNN.
Multi-scale features from both CNN stages are concatenated to form a per-channel embedding.
A Transformer encoder then models inter-channel spectral relationships.

<img width="615" height="1262" alt="CNN-Transformer architecture" src="https://github.com/user-attachments/assets/9badb9ca-fac3-4736-8826-565a8bc140ef" />

### Model 3 — Fusion (LSTM + CNN → Self-Attention)

Both temporal and spectral branches run in parallel on the same EEG recording.
Per-channel features from each branch are concatenated, projected to a shared embedding
space, and refined by a multi-head self-attention layer before classification.

<img width="849" height="947" alt="Fusion model architecture" src="https://github.com/user-attachments/assets/6fd60009-462f-42c2-9688-3ab7aa405d8b" />

---

## Repository Layout

```
.
├── data/
│   ├── preprocessing.py   # Bandpass filter (0.5–30 Hz) and STFT spectrogram
│   ├── targets.py         # Majority-consensus label aggregation
│   └── dataset.py         # CNNDataset, LSTMDataset, FusionDataset (PyTorch)
├── models/
│   ├── cnn_transformer.py # CNNTransformerClassifier
│   ├── lstm_transformer.py# LSTMTransformerClassifier
│   └── fusion.py          # FusionClassifier
├── utils/
│   └── config.py          # Central hyperparameter dictionary
├── train.py               # Unified training entry point (argparse)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

The dataset expects `.parquet` files with one row per EEG sample (columns = electrode names)
and a CSV metadata file with `eeg_id` and `expert_consensus` columns.

```
data_dir/
    12345.parquet
    12346.parquet
    ...
train.csv   # columns: eeg_id, expert_consensus
```

### 3. Train a model

```bash
# CNN-Transformer
python train.py --model cnn \
    --data_dir /path/to/eeg_parquets \
    --target_csv /path/to/train.csv

# LSTM-Transformer
python train.py --model lstm \
    --data_dir /path/to/eeg_parquets \
    --target_csv /path/to/train.csv

# Fusion
python train.py --model fusion \
    --data_dir /path/to/eeg_parquets \
    --target_csv /path/to/train.csv \
    --epochs 300
```

The best checkpoint (by validation accuracy) is saved to `checkpoints/<model>_best.pth`.

### Optional overrides

```
--epochs      INT    Number of training epochs
--lr          FLOAT  Initial learning rate
--batch_size  INT    Batch size
--num_workers INT    DataLoader workers (default 4)
--output_dir  PATH   Checkpoint output directory (default "checkpoints")
```

---

## Signal Processing

| Step | Setting |
|---|---|
| Window | Central 10 s (2 000 samples at 200 Hz) |
| Filter | 4th-order Butterworth bandpass 0.5–30 Hz |
| Normalisation | StandardScaler (zero mean, unit variance per recording) |
| Spectrogram | STFT — nperseg=64, 50 % overlap, magnitude |
| Sliding windows | 1 s windows (200 samples), 50-sample hop |
| Channels | 19 referential electrodes (EKG excluded) |

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam, lr=1e-4, weight_decay=1e-3 |
| Scheduler | OneCycleLR (max_lr=5e-3, 15-epoch warmup) |
| Gradient clipping | 1.0 |
| Class imbalance | WeightedRandomSampler |
| Validation split | 20 % stratified |

---

## Reference

> **Seizure Detection from EEG Signals using Multimodal Deep Learning**  
> S. H. Mirhoseini, 2024  
> Binary classification: Seizure vs LPD — 19-channel referential EEG, 200 Hz sampling rate
