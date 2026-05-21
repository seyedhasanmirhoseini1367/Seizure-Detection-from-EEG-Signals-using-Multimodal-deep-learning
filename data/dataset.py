"""
PyTorch Dataset classes for EEG seizure / LPD classification.

Three datasets are provided, each tailored to one model architecture:

    CNNDataset    — returns per-channel STFT spectrograms for CNN-Transformer.
    LSTMDataset   — returns raw temporal signals for LSTM-Transformer.
    FusionDataset — returns both modalities for the Fusion model.

All datasets read 10-second EEG windows centred on each recording, apply
bandpass filtering and StandardScaler normalisation per recording, and encode
binary labels (Seizure=0, LPD=1 by default via LabelEncoder).
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

from data.preprocessing import bandpass_filter, create_stft_spectrogram


# ── Constants ──────────────────────────────────────────────────────────────────

CLASSES = ["Seizure", "LPD"]
NUM_CHANNELS = 19       # referential EEG channels after dropping EKG
SAMPLING_RATE = 200     # Hz
WINDOW_SAMPLES = 10 * SAMPLING_RATE  # 2 000 samples = 10 s


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_window(
    file_path: str,
    window_samples: int = WINDOW_SAMPLES,
) -> Optional[pd.DataFrame]:
    """
    Load the central ``window_samples`` rows from a parquet EEG file.

    Returns ``None`` if the file is missing, too short, or cannot be read.
    """
    try:
        data = pd.read_parquet(file_path).drop(columns=["EKG"], errors="ignore")
        center = len(data) // 2
        half = window_samples // 2
        start, end = center - half, center + half
        if end - start < window_samples:
            warnings.warn(f"File too short: {file_path}")
            return None
        return data.iloc[start:end, :]
    except Exception as exc:
        warnings.warn(f"Could not read {file_path}: {exc}")
        return None


def _filter_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Bandpass-filter every column then standardise with zero mean / unit std."""
    filtered = pd.DataFrame(
        {col: bandpass_filter(df[col].values) for col in df.columns},
        index=df.index,
    )
    scaled = StandardScaler().fit_transform(filtered)
    return pd.DataFrame(scaled, columns=filtered.columns)


class _BaseEEGDataset(Dataset):
    """Shared initialisation logic for all three EEG dataset classes."""

    def __init__(
        self,
        ids: list[str],
        base_path: str,
        target_path: str,
        transform=None,
    ) -> None:
        self.id_list = ids
        self.base_path = base_path
        self.transform = transform

        label_encoder = LabelEncoder()
        label_encoder.fit(CLASSES)

        df = pd.read_csv(target_path)
        labels_df = df[df["target"].isin(CLASSES)]
        self.labels: dict[str, int] = dict(
            zip(
                labels_df["eeg_id"].astype(str),
                label_encoder.transform(labels_df["target"]),
            )
        )

    def __len__(self) -> int:
        return len(self.id_list)

    def get_class_weights(self) -> torch.Tensor:
        """Return inverse-frequency class weights for weighted sampling / loss."""
        labels = np.array([self.labels[str(i)] for i in self.id_list])
        counts = np.bincount(labels, minlength=len(CLASSES))
        weights = len(labels) / (len(CLASSES) * counts.clip(min=1))
        return torch.FloatTensor(weights)


# ── CNNDataset ─────────────────────────────────────────────────────────────────

class CNNDataset(_BaseEEGDataset):
    """
    Spectral dataset for the CNN-Transformer model.

    Each sample is a stack of per-channel STFT spectrograms of shape
    ``(num_channels, n_freqs, n_frames)``.

    Parameters
    ----------
    ids         : List of EEG recording IDs to include.
    base_path   : Directory containing ``<eeg_id>.parquet`` files.
    target_path : Path to the CSV file with ``eeg_id`` and ``target`` columns.
    nperseg     : STFT segment length in samples (default 64).
    noverlap    : STFT overlap; defaults to 50 % of ``nperseg``.
    transform   : Optional callable applied to the spectrogram tensor.
    """

    def __init__(
        self,
        ids: list[str],
        base_path: str,
        target_path: str,
        nperseg: int = 64,
        noverlap: Optional[int] = None,
        transform=None,
    ) -> None:
        super().__init__(ids, base_path, target_path, transform)
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2

    def __getitem__(self, index: int):
        file_id = str(self.id_list[index])
        file_path = os.path.join(self.base_path, f"{file_id}.parquet")

        window = _load_window(file_path)
        if window is None:
            return None

        df_scaled = _filter_and_scale(window)

        spectrograms = [
            create_stft_spectrogram(
                df_scaled.iloc[:, i].values,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
            )
            for i in range(df_scaled.shape[1])
        ]

        tensor = torch.tensor(np.stack(spectrograms, axis=0), dtype=torch.float32)
        if self.transform:
            tensor = self.transform(tensor)

        label = self.labels[file_id]
        return tensor, label


# ── LSTMDataset ────────────────────────────────────────────────────────────────

class LSTMDataset(_BaseEEGDataset):
    """
    Temporal dataset for the LSTM-Transformer model.

    Each sample is a per-channel sliding-window representation of shape
    ``(num_channels, n_windows, window_size)``.

    Parameters
    ----------
    ids         : List of EEG recording IDs.
    base_path   : Directory containing ``<eeg_id>.parquet`` files.
    target_path : Path to the CSV with ``eeg_id`` and ``target`` columns.
    window_size : Number of samples per sliding window (default 200 = 1 s).
    step_size   : Hop between consecutive windows in samples (default 50).
    transform   : Optional callable applied to the signal tensor.
    """

    def __init__(
        self,
        ids: list[str],
        base_path: str,
        target_path: str,
        window_size: int = 200,
        step_size: int = 50,
        transform=None,
    ) -> None:
        super().__init__(ids, base_path, target_path, transform)
        self.window_size = window_size
        self.step_size = step_size

    def _sliding_window(self, signal: np.ndarray) -> np.ndarray:
        """Return an array of shape ``(n_windows, window_size)``."""
        starts = range(0, len(signal) - self.window_size + 1, self.step_size)
        return np.array([signal[i: i + self.window_size] for i in starts])

    def __getitem__(self, index: int):
        file_id = str(self.id_list[index])
        file_path = os.path.join(self.base_path, f"{file_id}.parquet")

        window = _load_window(file_path)
        if window is None:
            return None

        df_scaled = _filter_and_scale(window)

        windowed = [
            self._sliding_window(df_scaled.iloc[:, i].values)
            for i in range(df_scaled.shape[1])
        ]

        tensor = torch.tensor(np.stack(windowed, axis=0), dtype=torch.float32)
        if self.transform:
            tensor = self.transform(tensor)

        label = self.labels[file_id]
        return tensor, label


# ── FusionDataset ──────────────────────────────────────────────────────────────

class FusionDataset(_BaseEEGDataset):
    """
    Multimodal dataset for the Fusion model.

    Returns both a spectrogram tensor and a temporal sliding-window tensor so
    that both CNN and LSTM branches receive their respective inputs.

    Parameters
    ----------
    ids         : List of EEG recording IDs.
    base_path   : Directory containing ``<eeg_id>.parquet`` files.
    target_path : Path to the CSV with ``eeg_id`` and ``target`` columns.
    nperseg     : STFT segment length in samples (default 64).
    noverlap    : STFT overlap; defaults to 50 % of ``nperseg``.
    window_size : Sliding-window size for the LSTM branch (default 200).
    step_size   : Hop between windows (default 50).
    transform   : Optional callable applied to both tensors.
    """

    def __init__(
        self,
        ids: list[str],
        base_path: str,
        target_path: str,
        nperseg: int = 64,
        noverlap: Optional[int] = None,
        window_size: int = 200,
        step_size: int = 50,
        transform=None,
    ) -> None:
        super().__init__(ids, base_path, target_path, transform)
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.window_size = window_size
        self.step_size = step_size

    def _sliding_window(self, signal: np.ndarray) -> np.ndarray:
        starts = range(0, len(signal) - self.window_size + 1, self.step_size)
        return np.array([signal[i: i + self.window_size] for i in starts])

    def __getitem__(self, index: int):
        file_id = str(self.id_list[index])
        file_path = os.path.join(self.base_path, f"{file_id}.parquet")

        window = _load_window(file_path)
        if window is None:
            return None

        df_scaled = _filter_and_scale(window)

        spectrograms, windowed = [], []
        for i in range(df_scaled.shape[1]):
            ch = df_scaled.iloc[:, i].values
            spectrograms.append(
                create_stft_spectrogram(ch, nperseg=self.nperseg, noverlap=self.noverlap)
            )
            windowed.append(self._sliding_window(ch))

        cnn_tensor = torch.tensor(np.stack(spectrograms, axis=0), dtype=torch.float32)
        lstm_tensor = torch.tensor(np.stack(windowed, axis=0), dtype=torch.float32)

        if self.transform:
            cnn_tensor = self.transform(cnn_tensor)
            lstm_tensor = self.transform(lstm_tensor)

        label = self.labels[file_id]
        return cnn_tensor, lstm_tensor, label
