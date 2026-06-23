#!/usr/bin/env python3
"""
Waterline Simulation - Test operational training across months

Simulates monthly model retraining from a start date to see how
R² varies across seasons with the waterline approach.

Now includes hyperparameter tuning for each simulation run to ensure
optimal model performance.

Usage:
    # Module-style invocation
    python -m training.prod.simulation --do_tuning --tuning_trials 15

    # Without tuning (faster, uses default params)
    python -m training.prod.simulation --no_tuning
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.utils.trainer import Trainer


def run_waterline_simulation(
    start_date: str = "2025-01-01",
    end_date: str = "2025-06-01",
    cutoff_days: int = 90,
    validation_days: int = 90,
    do_tuning: bool = True,
    tuning_trials: int = 15,
):
    """
    Run operational training for each month from start_date to end_date.

    This simulates what would happen if we deployed monthly retraining
    starting from start_date.

    Args:
        start_date: First reference date to simulate
        end_date: Last reference date to simulate
        cutoff_days: Days before reference_date for waterline
        validation_days: Size of validation window
        do_tuning: Whether to run hyperparameter tuning for each simulation
        tuning_trials: Number of tuning trials per simulation
    """

    repo_root = Path(__file__).parent.parent
    master_lake = repo_root / "feature_engineering" / "output" / "master_lake" / "master"

    trainer = Trainer(
        master_lake_path=str(master_lake),
        registry_csv=str(repo_root / "training" / "data" / "features_registry.csv"),
        results_dir=str(repo_root / "training" / "results" / "waterline_simulation"),
        use_latlon=False,
        strict_registry=True,
    )

    # Generate monthly reference dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    results = []
    current = start

    while current <= end:
        ref_date = current.strftime("%Y-%m-%d")
        # Match production.py logic exactly:
        # waterline = reference_date - cutoff_days
        # val_end = waterline
        # val_start = waterline - validation_days
        waterline_date = current - timedelta(days=cutoff_days)
        val_end_date = waterline_date
        val_start_date = waterline_date - timedelta(days=validation_days)

        waterline = waterline_date.strftime("%Y-%m-%d")
        val_end = val_end_date.strftime("%Y-%m-%d")
        val_start = val_start_date.strftime("%Y-%m-%d")

        print("\n" + "=" * 70)
        print(f"SIMULATION: Reference Date = {ref_date}")
        print(f"  Waterline (W):       {waterline}")
        print(f"  Training cutoff:     < {val_start}")
        print(f"  Validation window:   {val_start} to {val_end} ({validation_days} days)")
        print("=" * 70)

        try:
            result = trainer.run_operational_training(
                reference_date=ref_date,
                cutoff_days=cutoff_days,
                validation_days=validation_days,
                do_tuning=do_tuning,
                tuning_trials=tuning_trials,
            )

            metrics = result["val_metrics"]
            results.append({
                "reference_date": ref_date,
                "waterline": waterline,
                "val_start": val_start,
                "val_end": val_end,
                # Legacy metrics
                "r2": metrics.get("r2", 0),
                "rmse": metrics.get("rmse", 0),
                "mae": metrics.get("mae", 0),
                "bias": metrics.get("bias", 0),
                # MAE-skill metrics (Mason 2004)
                "mae_station_balanced": metrics.get("mae_station_balanced"),
                "mae_baseline_station_month": metrics.get("mae_baseline_station_month"),
                "mae_baseline_yesterday": metrics.get("mae_baseline_yesterday"),
                "skill_vs_station_month": metrics.get("skill_vs_station_month"),
                "skill_vs_yesterday": metrics.get("skill_vs_yesterday"),
                "skill_balanced_vs_station_month": metrics.get("skill_balanced_vs_station_month"),
                "n_stations": metrics.get("n_stations"),
                "n_train": result.get("n_train", 0),
                "n_val": result.get("n_val", 0),
                "status": "success",
            })

            skill_sm = metrics.get('skill_vs_station_month', 0) or 0
            skill_yd = metrics.get('skill_vs_yesterday', 0) or 0
            print(f"\n  ✓ MAE = {metrics.get('mae', 0):.1f}, Skill(s-m) = {skill_sm:.3f}, Skill(yd) = {skill_yd:.3f}")

        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            results.append({
                "reference_date": ref_date,
                "waterline": waterline,
                "val_start": val_start,
                "val_end": val_end,
                "status": f"failed: {e}",
            })

        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    # Summary
    print("\n" + "=" * 70)
    print("WATERLINE SIMULATION SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)

    if "mae" in df.columns:
        print(f"\n{'Ref Date':<12} {'Waterline':<12} {'MAE':>8} {'Skill(s-m)':>12} {'Skill(yd)':>12} {'R²':>8}")
        print("-" * 75)

        for _, row in df.iterrows():
            if row.get("status") == "success":
                mae_val = row.get('mae', 0) or 0
                skill_sm = row.get('skill_vs_station_month', 0) or 0
                skill_yd = row.get('skill_vs_yesterday', 0) or 0
                r2_val = row.get('r2', 0) or 0
                print(f"{row['reference_date']:<12} {row['waterline']:<12} {mae_val:>8.1f} {skill_sm:>12.3f} {skill_yd:>12.3f} {r2_val:>8.3f}")
            else:
                print(f"{row['reference_date']:<12} {row['waterline']:<12} {'FAILED':<25}")

        print("-" * 75)
        successful = df[df["status"] == "success"]
        if len(successful) > 0:
            print(f"\nMAE-SKILL STATISTICS (primary metrics):")
            print(f"  MAE:              {successful['mae'].mean():.1f} ± {successful['mae'].std():.1f} μg/m³")
            if 'skill_vs_station_month' in successful.columns:
                skill_sm = successful['skill_vs_station_month'].dropna()
                if len(skill_sm) > 0:
                    print(f"  Skill vs s-month: {skill_sm.mean():.3f} ± {skill_sm.std():.3f}")
            if 'skill_vs_yesterday' in successful.columns:
                skill_yd = successful['skill_vs_yesterday'].dropna()
                if len(skill_yd) > 0:
                    print(f"  Skill vs yesterday: {skill_yd.mean():.3f} ± {skill_yd.std():.3f}")
            print(f"\nLEGACY METRICS:")
            print(f"  R²:   {successful['r2'].mean():.3f} ± {successful['r2'].std():.3f}")
            print(f"  RMSE: {successful['rmse'].mean():.1f} ± {successful['rmse'].std():.1f} μg/m³")

    # Save results
    output_path = repo_root / "training" / "results" / "waterline_simulation" / "simulation_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Waterline simulation with hyperparameter tuning")
    parser.add_argument("--start", default="2025-01-01", help="Start reference date")
    parser.add_argument("--end", default="2025-06-01", help="End reference date")
    parser.add_argument("--cutoff", type=int, default=90, help="Cutoff days")
    parser.add_argument("--validation", type=int, default=90, help="Validation days")
    parser.add_argument("--no_tuning", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--do_tuning", action="store_true", help="Enable hyperparameter tuning (default)")
    parser.add_argument("--tuning_trials", type=int, default=15, help="Tuning trials per simulation (default: 15)")

    args = parser.parse_args()

    # Determine tuning behavior
    do_tuning = True  # default is to tune
    if args.no_tuning:
        do_tuning = False
    elif args.do_tuning:
        do_tuning = True

    run_waterline_simulation(
        start_date=args.start,
        end_date=args.end,
        cutoff_days=args.cutoff,
        validation_days=args.validation,
        do_tuning=do_tuning,
        tuning_trials=args.tuning_trials,
    )
