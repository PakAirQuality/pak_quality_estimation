"""
CLI entry point for U-Net training (v2).

Usage:
    python training/models/unet/run.py [OPTIONS]

Example:
    python training/models/unet/run.py \
        --grid_store inference/derived/feature_store/grid \
        --master_lake feature_engineering/output/master_lake/master \
        --results_dir training/results/unet_v2 \
        --epochs 100 \
        --batch_size 4 \
        --lr 1e-3 \
        --lambda_tv 0.01

Note: Runs training inline (not via module import) to avoid a PyTorch/pyarrow
      segfault triggered by torch.stack inside imported module functions.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# MPS lacks grid_sampler_2d_backward; fall back to CPU for that single op
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Ensure project root is on sys.path when running as a script
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net for gridded PM2.5 estimation (v2)"
    )

    # Paths
    parser.add_argument("--grid_store", default="inference/derived/feature_store/grid")
    parser.add_argument("--master_lake", default="feature_engineering/output/master_lake/master")
    parser.add_argument("--results_dir", default="training/results/unet_v3")

    # Date splits
    parser.add_argument("--train_start", default="2024-01-01")
    parser.add_argument("--train_end", default="2024-09-30")
    parser.add_argument("--val_start", default="2024-10-01")
    parser.add_argument("--val_end", default="2024-12-31")
    parser.add_argument("--test_start", default="2025-01-01")
    parser.add_argument("--test_end", default="2025-07-01")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_tv", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="Device (auto-detect if empty)")

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    # --- Imports (deferred to avoid module-level torch/pyarrow interaction) ---
    from training.models.unet.dataset import N_INPUT_CHANNELS, GridPM25Dataset
    from training.models.unet.loss import StationListLoss
    from training.models.unet.model import SmallUNet
    from training.utils.metrics import evaluate_predictions

    # --- Helpers ---
    def date_range(start, end):
        return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]

    def filter_dates(dates, grid_store):
        return [d for d in dates if (grid_store / "met" / f"date={d}").exists()]

    def iter_batches(dataset, batch_size, shuffle=False):
        """Yield batches as lists of samples (no stacking — variable-length stations)."""
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            samples = [dataset[int(i)] for i in batch_idx]
            # Stack fixed-size tensors, keep station lists as lists
            yield {
                "X": torch.stack([s["X"] for s in samples]),
                "border": torch.stack([s["border"] for s in samples]),
                "station_grid": [s["station_grid"] for s in samples],
                "station_pm25": [s["station_pm25"] for s in samples],
            }

    def _sample_at_stations(pred, station_grids):
        """Bilinear-sample predicted map at station coords. Returns list of [N_i] arrays."""
        results = []
        for b in range(pred.shape[0]):
            coords = station_grids[b]
            if len(coords) == 0:
                results.append(np.array([], dtype=np.float32))
                continue
            grid = coords.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
            sampled = torch.nn.functional.grid_sample(
                pred[b : b + 1], grid,
                mode="bilinear", align_corners=True, padding_mode="zeros",
            )
            results.append(sampled[0, 0, :, 0].numpy())
        return results

    def eval_mae(model, dataset, batch_size, device):
        """Compute MAE at station locations on a dataset."""
        model.eval()
        all_errors = []
        with torch.no_grad():
            for batch in iter_batches(dataset, batch_size):
                pred = model(batch["X"].to(device)).cpu()
                pred_vals = _sample_at_stations(pred, batch["station_grid"])
                for b in range(pred.shape[0]):
                    obs = batch["station_pm25"][b].numpy()
                    if len(obs) == 0:
                        continue
                    all_errors.append(np.abs(pred_vals[b] - obs))
        if not all_errors:
            return float("inf")
        return float(np.concatenate(all_errors).mean())

    def eval_full(model, dataset, batch_size, device):
        """Full evaluation at station locations on a dataset."""
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in iter_batches(dataset, batch_size):
                pred = model(batch["X"].to(device)).cpu()
                pred_vals = _sample_at_stations(pred, batch["station_grid"])
                for b in range(pred.shape[0]):
                    obs = batch["station_pm25"][b].numpy()
                    if len(obs) == 0:
                        continue
                    all_pred.append(pred_vals[b])
                    all_true.append(obs)
        if not all_true:
            return {}
        yt = np.concatenate(all_true)
        yp = np.concatenate(all_pred)
        print(f"  Test predictions: {len(yt)} station-day observations")
        return evaluate_predictions(yt, yp, verbose=True)

    # --- Setup ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    grid_store = Path(args.grid_store)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Date splits ---
    train_dates = filter_dates(date_range(args.train_start, args.train_end), grid_store)
    val_dates = filter_dates(date_range(args.val_start, args.val_end), grid_store)
    test_dates = filter_dates(date_range(args.test_start, args.test_end), grid_store)

    print(f"Train dates: {len(train_dates)}, Val dates: {len(val_dates)}, Test dates: {len(test_dates)}")

    if not train_dates:
        raise RuntimeError("No training dates found — check grid_store path.")

    # --- Datasets ---
    print("Building training dataset (computing normalization stats)...")
    train_ds = GridPM25Dataset(train_dates, grid_store, args.master_lake)
    norm_stats = train_ds.get_norm_stats_dict()

    with open(results_dir / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    nsd = {"mean": norm_stats["mean"], "std": norm_stats["std"]}
    val_ds = GridPM25Dataset(val_dates, grid_store, args.master_lake, norm_stats=nsd)
    test_ds = GridPM25Dataset(test_dates, grid_store, args.master_lake, norm_stats=nsd)

    n_train_batches = (len(train_ds) + args.batch_size - 1) // args.batch_size

    # --- Model ---
    model = SmallUNet(in_channels=N_INPUT_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = StationListLoss(lambda_tv=args.lambda_tv)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    best_val_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_mae": [], "lr": []}

    print(f"\nStarting training for up to {args.epochs} epochs "
          f"({n_train_batches} batches/epoch)...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in iter_batches(train_ds, args.batch_size, shuffle=True):
            X = batch["X"].to(device)
            border = batch["border"].to(device)

            pred = model(X)
            loss = criterion(
                pred,
                batch["station_grid"],
                batch["station_pm25"],
                border,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # --- Validate ---
        val_mae = eval_mae(model, val_ds, args.batch_size, device)

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

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

        gc.collect()

    elapsed = time.time() - t_start
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best val MAE: {best_val_mae:.1f}")

    # --- Test evaluation ---
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(results_dir / "best_model.pt", weights_only=True))
    model.to(device)

    test_metrics = eval_full(model, test_ds, args.batch_size, device)

    # --- Save results ---
    results = {
        "config": {
            "grid_store": args.grid_store,
            "master_lake": args.master_lake,
            "train_dates": len(train_dates),
            "val_dates": len(val_dates),
            "test_dates": len(test_dates),
            "epochs_run": len(history["train_loss"]),
            "n_channels": N_INPUT_CHANNELS,
            "n_params": n_params,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_tv": args.lambda_tv,
            "patience": args.patience,
            "seed": args.seed,
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

    print("\n=== Final Test Metrics ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
