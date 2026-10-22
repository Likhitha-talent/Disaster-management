"""
Command-Line Interface for Training Disaster Assessment Models
Supports training Stage A (BuildingLocalizationUNet) and Stage B (SiameseDamageUNet).
"""

import os
import sys
import argparse
import json

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from torch.utils.data import DataLoader
from ml.src.data.dataset import (
    xBDDisasterDataset,
    scan_xbd_directory,
    create_reproducible_splits,
    create_verified_benchmark_subset,
    DEFAULT_DATA_DIR
)
from ml.src.training.trainer import (
    TrainingConfig,
    LocalizationTrainer,
    DamageTrainer,
    DEFAULT_CHECKPOINT_DIR
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Disaster Damage Assessment ML Models")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Root directory containing disaster dataset (images/ and labels/)")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory to save model checkpoints")
    parser.add_argument("--model", type=str, choices=["localization", "damage", "both"], default="damage",
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile resolution")
    parser.add_argument("--measure-calibration", action="store_true", default=True,
                        help="Compute ECE calibration on validation split")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Explicitly allow overwriting existing checkpoint files")
    parser.add_argument("--create-benchmark-if-empty", action="store_true", default=True,
                        help="Auto-construct 24-pair benchmark subset on Drive D: if directory is empty")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("  AI-POWERED POST-DISASTER DAMAGE MODEL TRAINING PIPELINE")
    print(f"  Target: {args.model.upper()} | Device: {'CUDA' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'CPU'}")
    print(f"  Dataset: {args.data_dir} | Checkpoints: {args.checkpoint_dir}")
    print("=" * 70)

    # 1. Scan dataset
    samples = scan_xbd_directory(args.data_dir)
    if len(samples) == 0:
        if args.create_benchmark_if_empty:
            print(f"[i] No existing xBD image pairs found in '{args.data_dir}'.")
            print(f"[*] Constructing verified benchmark subset (24 paired disaster tiles)...")
            samples = create_verified_benchmark_subset(target_dir=args.data_dir, num_pairs=24, tile_size=args.tile_size, seed=args.seed)
            print(f"[OK] Benchmark dataset ready: {len(samples)} pairs created in {args.data_dir}.")
        else:
            print(f"[ERROR] No paired xBD satellite images found in '{args.data_dir}'. Aborting.")
            sys.exit(1)

    print(f"\n[1/3] Partitioning {len(samples)} image pairs into reproducible Train/Val/Test splits...")
    splits = create_reproducible_splits(samples, output_dir=args.checkpoint_dir, seed=args.seed)
    print(f"      Train: {len(splits['train'])} | Validation: {len(splits['val'])} | Test: {len(splits['test'])}")

    # 2. Build DataLoaders
    train_ds = xBDDisasterDataset(splits["train"], tile_size=args.tile_size, augment=True, seed=args.seed)
    val_ds = xBDDisasterDataset(splits["val"], tile_size=args.tile_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = TrainingConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        tile_size=args.tile_size,
        seed=args.seed,
        measure_calibration=args.measure_calibration,
        overwrite_checkpoints=args.overwrite
    )

    dataset_info = {
        "dataset_dir": args.data_dir,
        "total_pairs": len(samples),
        "split_counts": {"train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"])},
        "tile_size": args.tile_size
    }

    # 3. Execute Training
    results = {}

    if args.model in ["localization", "both"]:
        print(f"\n[2/3] Training BuildingLocalizationUNet ({args.epochs} epochs)...")
        loc_trainer = LocalizationTrainer(config)
        loc_res = loc_trainer.fit(train_loader, val_loader, dataset_info)
        results["localization"] = loc_res
        print(f"      [OK] Localization Best Val IoU: {loc_res['best_iou']:.4f}")
        print(f"      Checkpoint saved to: {loc_res['checkpoint_path']}")

    if args.model in ["damage", "both"]:
        print(f"\n[3/3] Training SiameseDamageUNet ({args.epochs} epochs)...")
        dmg_trainer = DamageTrainer(config)
        dmg_res = dmg_trainer.fit(train_loader, val_loader, dataset_info)
        results["damage"] = dmg_res
        print(f"      [OK] Damage Best Harmonic F1: {dmg_res['best_harmonic_f1']:.4f}")
        print(f"      Checkpoint saved to: {dmg_res['checkpoint_path']}")

        cal = dmg_res["best_metrics"].get("calibration")
        if cal:
            print(f"      [i] Empirical Calibration ECE: {cal['ece']:.4f} (Certified: {cal['is_calibrated']})")

    # Save summary report
    report_path = os.path.join(args.checkpoint_dir, "training_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETE")
    print(f"  Summary Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

