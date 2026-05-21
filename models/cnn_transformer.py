"""
CNN-Transformer Classifier for EEG seizure detection.

Architecture
------------
Each EEG channel's STFT spectrogram is processed independently through a
two-stage CNN.  Multi-scale features from both stages are concatenated to form
a per-channel embedding.  A Transformer encoder then models inter-channel
relationships before a three-layer MLP produces the final classification.

    Input  : (B, C, F, T)  — batch of C-channel spectrograms
    Output : (B, num_classes)

Reference
---------
Thesis: "Seizure Detection from EEG Signals using Multimodal Deep Learning"
Model 2 — CNN → Transformer  |  Accuracy: 90.12 %
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNTransformerClassifier(nn.Module):
    """
    Spectral feature extractor with a Transformer classification head.

    Parameters
    ----------
    in_channels        : Channels going into the first conv block (default 1,
                         since each EEG channel is treated as a grayscale image).
    out_channels       : Number of filters for the two CNN blocks
                         (default ``[64, 64]``).
    num_classes        : Number of output classes (default 2).
    dropout            : Dropout probability (default 0.5).
    transformer_layers : Number of Transformer encoder layers (default 2).
    nhead              : Number of attention heads (default 8).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: list[int] | None = None,
        num_classes: int = 2,
        dropout: float = 0.5,
        transformer_layers: int = 2,
        nhead: int = 8,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = [64, 64]

        d_model = sum(out_channels)  # concatenated multi-scale feature size

        # ── CNN blocks ────────────────────────────────────────────────────
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=8, padding=3),
            nn.BatchNorm2d(out_channels[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=6, padding=2),
            nn.BatchNorm2d(out_channels[1]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Transformer encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, C, F, T)`` — batch of multi-channel
            STFT spectrograms.

        Returns
        -------
        Tensor of shape ``(B, num_classes)`` — class logits.
        """
        B, C, F, T = x.shape

        # Merge batch and channel dims so the CNN sees each channel independently
        x = x.view(B * C, 1, F, T)

        # Two-stage CNN with multi-scale pooling
        feat1 = self.conv_block1(x)                          # (B*C, out[0], F/2, T/2)
        feat2 = self.conv_block2(feat1)                      # (B*C, out[1], F/4, T/4)

        feat1 = self.pool(feat1).view(B * C, -1)             # (B*C, out[0])
        feat2 = self.pool(feat2).view(B * C, -1)             # (B*C, out[1])

        features = torch.cat([feat1, feat2], dim=1)          # (B*C, d_model)
        features = features.view(B, C, -1)                   # (B, C, d_model)

        # Transformer over channel tokens
        out = self.transformer(features)                     # (B, C, d_model)
        out = out.mean(dim=1)                                # (B, d_model)

        return self.classifier(out)                          # (B, num_classes)
