"""
train.py — Unified training script for EEG seizure detection models.

Supported models
----------------
  cnn      : CNN-Transformer  (spectrograms)
  lstm     : LSTM-Transformer (raw sliding windows)
  fusion   : Fusion model     (both modalities)

Usage examples
--------------
  python train.py --model cnn   --data_dir /data/eeg --target_csv /data/train.csv
  python train.py --model lstm  --data_dir /data/eeg --target_csv /data/train.csv
  python train.py --model fusion --data_dir /data/eeg --target_csv /data/train.csv --epochs 300
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import CNNDataset, FusionDataset, LSTMDataset
from data.preprocessing import list_file_ids, remove_nan_files
from data.targets import build_label_map
from models import CNNTransformerClassifier, FusionClassifier, LSTMTransformerClassifier
from utils.config import CONFIG

import pandas as pd


# ── Reproducibility ────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data loading ───────────────────────────────────────────────────────────────

def _make_loaders(args, cfg):
    """Build train/val DataLoaders for the requested model type."""
    df = pd.read_csv(args.target_csv)
    label_map = build_label_map(df)

    all_ids = list_file_ids(args.data_dir)
    all_ids, df = remove_nan_files(args.data_dir, all_ids, df)
    valid_ids = [i for i in all_ids if str(i) in label_map and label_map[str(i)] is not None]

    train_ids, val_ids = train_test_split(
        valid_ids,
        test_size=cfg["val_split"],
        random_state=cfg["random_seed"],
        stratify=[label_map[str(i)] for i in valid_ids],
    )

    ds_kwargs = dict(base_path=args.data_dir, target_path=args.target_csv)

    if args.model == "cnn":
        train_ds = CNNDataset(train_ids, nperseg=cfg["nperseg"], noverlap=cfg["noverlap"], **ds_kwargs)
        val_ds   = CNNDataset(val_ids,   nperseg=cfg["nperseg"], noverlap=cfg["noverlap"], **ds_kwargs)
    elif args.model == "lstm":
        train_ds = LSTMDataset(train_ids, window_size=cfg["window_size"], step_size=cfg["step_size"], **ds_kwargs)
        val_ds   = LSTMDataset(val_ids,   window_size=cfg["window_size"], step_size=cfg["step_size"], **ds_kwargs)
    else:  # fusion
        train_ds = FusionDataset(
            train_ids, nperseg=cfg["nperseg"], noverlap=cfg["noverlap"],
            window_size=cfg["window_size"], step_size=cfg["step_size"], **ds_kwargs,
        )
        val_ds = FusionDataset(
            val_ids, nperseg=cfg["nperseg"], noverlap=cfg["noverlap"],
            window_size=cfg["window_size"], step_size=cfg["step_size"], **ds_kwargs,
        )

    # Weighted sampler to handle class imbalance
    class_weights = train_ds.get_class_weights()
    sample_weights = [class_weights[train_ds.labels[str(i)]] for i in train_ids]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False,  **loader_kwargs)

    return train_loader, val_loader


# ── Model factory ──────────────────────────────────────────────────────────────

def _build_model(args, cfg) -> nn.Module:
    if args.model == "cnn":
        return CNNTransformerClassifier(
            out_channels=cfg["cnn_out_channels"],
            num_classes=cfg["num_classes"],
            dropout=cfg["cnn_dropout"],
            transformer_layers=cfg["cnn_transformer_layers"],
            nhead=cfg["cnn_nhead"],
        )
    elif args.model == "lstm":
        return LSTMTransformerClassifier(
            input_size=cfg["window_size"],
            lstm_hidden_size=cfg["lstm_hidden_size"],
            lstm_layers=cfg["lstm_layers"],
            transformer_heads=cfg["lstm_nhead"],
            transformer_layers=cfg["lstm_transformer_layers"],
            dropout_rate=cfg["lstm_dropout"],
            num_classes=cfg["num_classes"],
        )
    else:  # fusion
        return FusionClassifier(
            cnn_channels=cfg["fusion_cnn_channels"],
            lstm_input_size=cfg["window_size"],
            lstm_hidden_size=cfg["fusion_lstm_hidden"],
            d_model=cfg["fusion_d_model"],
            num_heads=cfg["fusion_nhead"],
            dropout=cfg["fusion_dropout"],
            num_classes=cfg["num_classes"],
        )


# ── Training loop ──────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, device, is_fusion: bool, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            if batch is None:
                continue

            if is_fusion:
                cnn_x, lstm_x, labels = batch
                cnn_x  = cnn_x.to(device)
                lstm_x = lstm_x.to(device)
                labels = labels.to(device)
                logits = model(cnn_x, lstm_x)
            else:
                x, labels = batch
                x      = x.to(device)
                labels = labels.to(device)
                logits = model(x)

            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train(args) -> None:
    cfg = {**CONFIG}
    # CLI overrides
    if args.epochs    is not None: cfg["epochs"]     = args.epochs
    if args.lr        is not None: cfg["lr"]         = args.lr
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size

    _seed_everything(cfg["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model.upper()}")

    train_loader, val_loader = _make_loaders(args, cfg)
    model = _build_model(args, cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["max_lr"],
        epochs=cfg["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=cfg["warmup_epochs"] / cfg["epochs"],
    )

    is_fusion  = args.model == "fusion"
    best_acc   = 0.0
    save_path  = os.path.join(args.output_dir, f"{args.model}_best.pth")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, is_fusion, train=True)
        scheduler.step()
        va_loss, va_acc = _run_epoch(model, val_loader,   criterion, None,      device, is_fusion, train=False)

        print(
            f"Epoch {epoch:03d}/{cfg['epochs']}  "
            f"train loss {tr_loss:.4f}  acc {tr_acc:.4f}  |  "
            f"val loss {va_loss:.4f}  acc {va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → saved best model  (val acc {best_acc:.4f})")

    print(f"\nTraining complete.  Best val accuracy: {best_acc:.4f}")
    print(f"Model saved to: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG seizure detection — model trainer")
    p.add_argument("--model",       required=True, choices=["cnn", "lstm", "fusion"])
    p.add_argument("--data_dir",    required=True, help="Directory of .parquet EEG files")
    p.add_argument("--target_csv",  required=True, help="CSV with eeg_id and expert_consensus columns")
    p.add_argument("--output_dir",  default="checkpoints",  help="Where to save best model weights")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--num_workers", type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
