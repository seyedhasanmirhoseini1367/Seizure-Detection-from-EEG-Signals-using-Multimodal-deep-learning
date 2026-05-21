"""
Central hyperparameter configuration.

All model-specific and training hyperparameters live here.
Override individual keys via argparse in train.py.
"""

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────
    "num_classes": 2,
    "classes": ["Seizure", "LPD"],
    "sampling_rate": 200,           # Hz
    "window_seconds": 10,           # seconds of EEG per sample
    "num_channels": 19,             # referential EEG channels (EKG excluded)

    # ── Signal preprocessing ──────────────────────────────────────────────
    "bandpass_lowcut": 0.5,         # Hz
    "bandpass_highcut": 30.0,       # Hz
    "bandpass_order": 4,

    # ── STFT spectrogram ──────────────────────────────────────────────────
    "nperseg": 64,                  # STFT window length (samples)
    "noverlap": 32,                 # 50 % overlap

    # ── LSTM sliding window ───────────────────────────────────────────────
    "window_size": 200,             # samples per window
    "step_size": 50,                # hop between windows

    # ── Training ──────────────────────────────────────────────────────────
    "batch_size": 32,
    "epochs": 200,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "warmup_epochs": 15,
    "max_lr": 5e-3,
    "min_lr": 1e-6,
    "grad_clip": 1.0,
    "val_split": 0.2,
    "random_seed": 42,

    # ── CNN-Transformer ───────────────────────────────────────────────────
    "cnn_out_channels": [64, 64],
    "cnn_nhead": 8,
    "cnn_transformer_layers": 2,
    "cnn_dropout": 0.5,

    # ── LSTM-Transformer ──────────────────────────────────────────────────
    "lstm_hidden_size": 16,
    "lstm_layers": 1,
    "lstm_nhead": 4,
    "lstm_transformer_layers": 2,
    "lstm_dropout": 0.5,

    # ── Fusion ────────────────────────────────────────────────────────────
    "fusion_cnn_channels": [64, 96],
    "fusion_lstm_hidden": 32,
    "fusion_d_model": 128,
    "fusion_nhead": 4,
    "fusion_dropout": 0.5,
}
