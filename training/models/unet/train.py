"""
U-Net Training Loop
===================

train_unet(config) handles:
  - Data split (train/val/test by date ranges)
  - Dataset construction with normalization
  - Training with AdamW + CosineAnnealingLR
  - Early stopping on validation MAE
  - Final test evaluation with training.utils.metrics
  - Artifact saving (model, metrics, norm stats)
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from training.utils.metrics import evaluate_predictions

from .dataset import ALL_FEATURES, GridPM25Dataset
from .loss import MaskedPM25Loss
from .model import SmallUNet


@dataclass
class UNetConfig:
    """Training configuration."""

    grid_store: str = "inference/derived/feature_store/grid"
    master_lake: str = "feature_engineering/output/master_lake/master"
    results_dir: str = "training/results/unet_v1"

    # Date splits
    train_start: str = "2024-01-01"
    train_end: str = "2024-09-30"
    val_start: str = "2024-10-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-07-01"

    # Training hyperparams
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_tv: float = 0.01
    patience: int = 15
    max_grad_norm: float = 1.0

    # Misc
    seed: int = 42
    device: str = ""  # auto-detect if empty


def train_unet(config: UNetConfig) -> Dict:
    """
    Full U-Net training pipeline.

    Returns dict with train/val/test metrics and config.
    """
    import pandas as pd

    sys.stdout.reconfigure(line_buffering=True)

    # --- Helpers ---
    def date_range(start: str, end: str) -> List[str]:
        return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]

    def filter_dates(dates: List[str]) -> List[str]:
        return [d for d in dates if (grid_store / "met" / f"date={d}").exists()]

    def collate(samples: list) -> Dict[str, torch.Tensor]:
        return {
            "X": torch.stack([s["X"] for s in samples]),
            "y": torch.stack([s["y"] for s in samples]),
            "mask": torch.stack([s["mask"] for s in samples]),
            "border": torch.stack([s["border"] for s in samples]),
        }

    def iter_batches(dataset, batch_size, shuffle=False):
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            samples = [dataset[int(i)] for i in batch_idx]
            yield collate(samples)

    def eval_mae(model, dataset, batch_size, device):
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in iter_batches(dataset, batch_size):
                pred = model(batch["X"].to(device)).cpu()
                m = batch["mask"].bool().flatten()
                if m.any():
                    all_true.append(batch["y"].flatten()[m].numpy())
                    all_pred.append(pred.flatten()[m].numpy())
        if not all_true:
            return float("inf")
        yt = np.concatenate(all_true)
        yp = np.concatenate(all_pred)
        return float(np.mean(np.abs(yt - yp)))

    def eval_full(model, dataset, batch_size, device):
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in iter_batches(dataset, batch_size):
                pred = model(batch["X"].to(device)).cpu()
                m = batch["mask"].bool().flatten()
                if m.any():
                    all_true.append(batch["y"].flatten()[m].numpy())
                    all_pred.append(pred.flatten()[m].numpy())
        if not all_true:
            return {}
        yt = np.concatenate(all_true)
        yp = np.concatenate(all_pred)
        print(f"  Test predictions: {len(yt)} station-day observations")
        return evaluate_predictions(yt, yp, verbose=True)

    # --- Setup ---
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.device:
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    grid_store = Path(config.grid_store)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Date splits ---
    train_dates = filter_dates(date_range(config.train_start, config.train_end))
    val_dates = filter_dates(date_range(config.val_start, config.val_end))
    test_dates = filter_dates(date_range(config.test_start, config.test_end))

    print(f"Train dates: {len(train_dates)}, Val dates: {len(val_dates)}, Test dates: {len(test_dates)}")

    if not train_dates:
        raise RuntimeError("No training dates found — check grid_store path.")

    # --- Datasets ---
    print("Building training dataset (computing normalization stats)...")
    train_ds = GridPM25Dataset(train_dates, grid_store, config.master_lake)
    norm_stats = train_ds.get_norm_stats_dict()

    with open(results_dir / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    nsd = {"mean": norm_stats["mean"], "std": norm_stats["std"]}
    val_ds = GridPM25Dataset(val_dates, grid_store, config.master_lake, norm_stats=nsd)
    test_ds = GridPM25Dataset(test_dates, grid_store, config.master_lake, norm_stats=nsd)

    n_train_batches = (len(train_ds) + config.batch_size - 1) // config.batch_size

    # --- Model ---
    n_channels = len(ALL_FEATURES)
    model = SmallUNet(in_channels=n_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = MaskedPM25Loss(lambda_tv=config.lambda_tv)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # --- Training loop ---
    best_val_mae = float("inf")
    patience_counter = 0
    history: Dict[str, list] = {"train_loss": [], "val_mae": [], "lr": []}

    print(f"\nStarting training for up to {config.epochs} epochs "
          f"({n_train_batches} batches/epoch)...")
    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in iter_batches(train_ds, config.batch_size, shuffle=True):
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            border = batch["border"].to(device)

            pred = model(X)
            loss = criterion(pred, y, mask, border)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # --- Validate ---
        val_mae = eval_mae(model, val_ds, config.batch_size, device)

        history["train_loss"].append(avg_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0:
            marker = " *" if patience_counter == 0 else ""
            elapsed_min = (time.time() - t_start) / 60
            print(
                f"  Epoch {epoch:3d} | loss={avg_loss:.2f} | val_MAE={val_mae:.1f}"
                f" | lr={optimizer.param_groups[0]['lr']:.2e}"
                f" | patience={patience_counter} | {elapsed_min:.1f}min{marker}"
            )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch} (patience={config.patience})")
            break

        gc.collect()

    elapsed = time.time() - t_start
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best val MAE: {best_val_mae:.1f}")

    # --- Test evaluation ---
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(results_dir / "best_model.pt", weights_only=True))
    model.to(device)

    test_metrics = eval_full(model, test_ds, config.batch_size, device)

    # --- Save results ---
    results = {
        "config": {
            "grid_store": config.grid_store,
            "master_lake": config.master_lake,
            "train_dates": len(train_dates),
            "val_dates": len(val_dates),
            "test_dates": len(test_dates),
            "epochs_run": len(history["train_loss"]),
            "n_channels": n_channels,
            "n_params": n_params,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "lambda_tv": config.lambda_tv,
            "patience": config.patience,
            "seed": config.seed,
            "device": str(device),
        },
        "best_val_mae": best_val_mae,
        "test": test_metrics,
        "history": history,
        "elapsed_minutes": elapsed / 60,
    }

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifacts saved to {results_dir}/")
    return results
