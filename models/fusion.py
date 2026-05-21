"""
Fusion Classifier for EEG seizure detection.

Architecture
------------
Two parallel branches encode complementary views of the same EEG recording:

  CNN branch  — processes per-channel STFT spectrograms (``B, C, F, T``)
                through a two-stage convolutional network.
  LSTM branch — processes per-channel raw sliding-window signals
                (``B, C, n_windows, window_size``) through a shared LSTM.

Per-channel features from both branches are concatenated and projected to a
common embedding space.  A single multi-head self-attention layer then models
inter-channel cross-modal relationships before a two-layer MLP produces the
final classification.

    CNN input  : (B, C, F, T)
    LSTM input : (B, C, n_windows, window_size)
    Output     : (B, num_classes)

Reference
---------
Thesis: "Seizure Detection from EEG Signals using Multimodal Deep Learning"
Model 3 — Fusion (CNN + LSTM → Self-Attention)  |  Accuracy: 82.51 %
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    """
    Multimodal EEG classifier combining spectral (CNN) and temporal (LSTM)
    features via cross-channel self-attention.

    Parameters
    ----------
    in_channels      : Input channels for each CNN conv block (default 1 —
                       each EEG channel treated as a grayscale spectrogram).
    cnn_channels     : Filter counts for the two CNN conv blocks
                       (default ``[64, 96]``).
    lstm_input_size  : Feature size per LSTM time step — equals ``window_size``
                       from the dataset (default 200).
    lstm_hidden_size : LSTM hidden-state dimension (default 32).
    d_model          : Embedding dimension after the fusion projection layer
                       (default 128).
    num_heads        : Number of self-attention heads (default 4).
    dropout          : Dropout probability (default 0.5).
    num_classes      : Number of output classes (default 2).
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: list[int] | None = None,
        lstm_input_size: int = 200,
        lstm_hidden_size: int = 32,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.5,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [64, 96]

        self.dropout = nn.Dropout(dropout)

        # ── CNN branch ────────────────────────────────────────────────────
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── LSTM branch ───────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=False,
        )

        # ── Fusion projection ─────────────────────────────────────────────
        fused_dim = cnn_channels[1] + lstm_hidden_size
        self.embedding = nn.Linear(fused_dim, d_model)

        # ── Self-attention over channel tokens ────────────────────────────
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        cnn_x: torch.Tensor,
        lstm_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cnn_x  : Tensor of shape ``(B, C, F, T)`` — per-channel STFT
                 spectrograms.
        lstm_x : Tensor of shape ``(B, C, n_windows, window_size)`` —
                 per-channel sliding-window signals.

        Returns
        -------
        Tensor of shape ``(B, num_classes)`` — class logits.
        """
        B, C, F, T = cnn_x.shape

        # ── CNN branch ────────────────────────────────────────────────────
        cnn_in = cnn_x.view(B * C, 1, F, T)             # (B*C, 1, F, T)
        cnn_feat = self.conv_layers(cnn_in)              # (B*C, ch[-1], f', t')
        cnn_feat = self.cnn_pool(cnn_feat)               # (B*C, ch[-1], 1, 1)
        cnn_feat = cnn_feat.view(B, C, -1)               # (B, C, ch[-1])

        # ── LSTM branch ───────────────────────────────────────────────────
        _, _, n_windows, window_size = lstm_x.shape
        lstm_in = lstm_x.view(B * C, n_windows, window_size)  # (B*C, n_win, win)
        lstm_out, _ = self.lstm(lstm_in)                       # (B*C, n_win, hid)
        lstm_feat = lstm_out[:, -1, :].view(B, C, -1)         # (B, C, hid)

        # ── Fusion ────────────────────────────────────────────────────────
        fused = torch.cat([cnn_feat, lstm_feat], dim=-1)  # (B, C, ch[-1]+hid)
        tokens = self.embedding(fused)                    # (B, C, d_model)

        # Self-attention aggregates cross-channel cross-modal context
        attn_out, _ = self.self_attention(tokens, tokens, tokens)  # (B, C, d)
        attn_out = self.layer_norm(attn_out.mean(dim=1))           # (B, d)
        attn_out = self.dropout(attn_out)

        return self.classifier(attn_out)                           # (B, num_classes)
