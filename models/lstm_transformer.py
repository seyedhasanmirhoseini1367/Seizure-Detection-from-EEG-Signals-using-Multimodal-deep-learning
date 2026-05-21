"""
LSTM-Transformer Classifier for EEG seizure detection.

Architecture
------------
Each EEG channel's sliding-window time series is encoded by a shared LSTM.
The per-channel last-step hidden states are stacked into a sequence and passed
to a Transformer encoder, which models temporal inter-channel relationships.
A two-layer MLP produces the final classification.

    Input  : (B, C, n_windows, window_size)  — windowed EEG signals
    Output : (B, num_classes)

Reference
---------
Thesis: "Seizure Detection from EEG Signals using Multimodal Deep Learning"
Model 1 — LSTM → Transformer  |  Accuracy: 79.15 %
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMTransformerClassifier(nn.Module):
    """
    Temporal feature extractor with a Transformer classification head.

    Each EEG channel's windowed signal is processed independently by an LSTM.
    The last hidden state from each channel is collected and treated as a
    sequence of channel tokens for the Transformer encoder.

    Parameters
    ----------
    input_size         : Feature size per time step — equals ``window_size``
                         from the dataset (default 200).
    lstm_hidden_size   : LSTM hidden-state size (adjusted to be divisible by
                         ``transformer_heads``; default 16).
    lstm_layers        : Number of stacked LSTM layers (default 1).
    transformer_heads  : Number of Transformer attention heads (default 4).
    transformer_layers : Number of Transformer encoder layers (default 2).
    dropout_rate       : Dropout probability (default 0.5).
    num_classes        : Number of output classes (default 2).
    """

    def __init__(
        self,
        input_size: int = 200,
        lstm_hidden_size: int = 16,
        lstm_layers: int = 1,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout_rate: float = 0.5,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        # Ensure hidden size is divisible by number of attention heads
        if lstm_hidden_size % transformer_heads != 0:
            lstm_hidden_size = (
                (lstm_hidden_size // transformer_heads) + 1
            ) * transformer_heads

        self.lstm_hidden_size = lstm_hidden_size
        d_model = lstm_hidden_size * 2  # doubled to provide richer embeddings

        # ── LSTM (shared across channels) ─────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )

        # ── Transformer encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=lstm_hidden_size * 4,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.dropout = nn.Dropout(dropout_rate)

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, C, n_windows, window_size)`` — per-channel
            sliding-window EEG signals.

        Returns
        -------
        Tensor of shape ``(B, num_classes)`` — class logits.
        """
        B, C, n_windows, window_size = x.shape

        # Encode each channel with the shared LSTM
        channel_embeddings = []
        h0 = torch.zeros(self.lstm.num_layers, B * C, self.lstm_hidden_size, device=x.device)
        c0 = torch.zeros_like(h0)

        # Merge batch and channel dims for parallel LSTM processing
        x_merged = x.view(B * C, n_windows, window_size)        # (B*C, n_windows, window_size)
        lstm_out, _ = self.lstm(x_merged, (h0, c0))             # (B*C, n_windows, hidden)
        last_hidden = lstm_out[:, -1, :]                         # (B*C, hidden)
        last_hidden = last_hidden.view(B, C, -1)                 # (B, C, hidden)

        # Duplicate hidden state to form d_model = 2 * hidden_size
        tokens = torch.cat([last_hidden, last_hidden], dim=-1)   # (B, C, d_model)

        # Transformer over channel tokens; use last token as summary
        out = self.transformer(tokens)                           # (B, C, d_model)
        out = out[:, -1, :]                                      # (B, d_model)

        out = self.dropout(out)
        return self.classifier(out)                              # (B, num_classes)
