"""
Signal preprocessing utilities for EEG seizure detection.

Functions
---------
bandpass_filter        : 4th-order Butterworth bandpass filter (zero-phase).
create_stft_spectrogram: Single-channel STFT spectrogram with log-scaling.
list_file_ids          : Scan a directory and return all numeric EEG file IDs.
remove_nan_files       : Remove parquet files that contain NaN values and return
                         the cleaned ID list and filtered metadata DataFrame.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, stft


# ── Bandpass filter ────────────────────────────────────────────────────────────

def _butter_bandpass(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return b, a


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 30.0,
    fs: float = 200.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to a 1-D signal.

    Parameters
    ----------
    signal  : 1-D array of EEG samples.
    lowcut  : Low cut-off frequency in Hz (default 0.5 Hz).
    highcut : High cut-off frequency in Hz (default 30 Hz).
    fs      : Sampling frequency in Hz (default 200 Hz).
    order   : Filter order (default 4).

    Returns
    -------
    Filtered signal as a numpy array of the same shape.
    """
    b, a = _butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)


# ── STFT spectrogram ───────────────────────────────────────────────────────────

def create_stft_spectrogram(
    signal: np.ndarray,
    sampling_rate: int = 200,
    nperseg: int = 64,
    noverlap: Optional[int] = None,
) -> np.ndarray:
    """
    Convert a single EEG channel to a normalised log-magnitude STFT spectrogram.

    Parameters
    ----------
    signal        : 1-D array of EEG samples.
    sampling_rate : Sampling rate in Hz (default 200 Hz).
    nperseg       : STFT segment length in samples (default 64).
    noverlap      : Overlap between segments; defaults to 50 % of ``nperseg``.

    Returns
    -------
    2-D array of shape (n_freqs, n_frames) with values in [0, ~log(2)].

    Raises
    ------
    ValueError : If the signal is shorter than ``nperseg``.
    """
    if len(signal) < nperseg:
        raise ValueError(
            f"Signal length {len(signal)} is shorter than nperseg={nperseg}."
        )

    if noverlap is None:
        noverlap = nperseg // 2

    _, _, Zxx = stft(signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

    magnitude = np.abs(Zxx)
    denom = magnitude.max() - magnitude.min()
    if denom > 0:
        magnitude = (magnitude - magnitude.min()) / denom  # min-max to [0, 1]
    return np.log1p(magnitude).astype(np.float32)           # log(1 + x)


# ── File utilities ─────────────────────────────────────────────────────────────

def list_file_ids(directory: str) -> list[str]:
    """
    Return all numeric file stems present in *directory*.

    For example, a directory containing ``12345.parquet`` yields ``["12345"]``.

    Parameters
    ----------
    directory : Path to the folder containing ``<eeg_id>.parquet`` files.

    Returns
    -------
    List of EEG ID strings (without extension).
    """
    return [
        os.path.splitext(fname)[0]
        for fname in os.listdir(directory)
        if os.path.splitext(fname)[0].isdigit()
    ]


def remove_nan_files(
    directory: str,
    id_list: list[str],
    df: pd.DataFrame,
    id_column: str = "eeg_id",
) -> tuple[list[str], pd.DataFrame]:
    """
    Scan EEG parquet files for NaN values, delete offending files, and return
    the cleaned ID list together with the correspondingly filtered DataFrame.

    Files whose ID is absent from ``df`` are also removed.

    Parameters
    ----------
    directory : Folder containing ``<eeg_id>.parquet`` files.
    id_list   : All file IDs present in ``directory``.
    df        : Metadata DataFrame with at least an ``id_column`` column.
    id_column : Name of the EEG-ID column in ``df`` (default ``"eeg_id"``).

    Returns
    -------
    valid_ids   : List of IDs whose files are clean and present in ``df``.
    filtered_df : ``df`` filtered to contain only rows with an ID in ``valid_ids``.
    """
    known_ids = set(df[id_column].astype(str))
    valid_ids: list[str] = []

    for file_id in id_list:
        file_path = os.path.join(directory, f"{file_id}.parquet")
        if file_id not in known_ids:
            _safe_remove(file_path)
            continue
        try:
            data = pd.read_parquet(file_path)
            if data.isna().any().any():
                _safe_remove(file_path)
            else:
                valid_ids.append(file_id)
        except Exception as exc:
            warnings.warn(f"Could not read {file_path}: {exc}")

    filtered_df = df[df[id_column].astype(str).isin(valid_ids)].reset_index(drop=True)
    return valid_ids, filtered_df


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
