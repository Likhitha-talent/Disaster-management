"""
Command-Line Interface for Independent Evaluation & Calibration Measurement
Evaluates trained checkpoints on held-out test splits and certifies ECE calibration.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Optional, Dict, Any
import torch
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from torch.utils.data import DataLoader
from ml.src.models.unet import BuildingLocalizationUNet
from ml.src.models.siamese_unet import SiameseDamageUNet
from ml.src.data.dataset import (
    xBDDisasterDataset,
    scan_xbd_directory,
    create_reproducible_splits,
    detect_dataset_origin,
    DEFAULT_DATA_DIR
)
from ml.src.training.trainer import CheckpointManager, DEFAULT_CHECKPOINT_DIR
from ml.src.training.metrics import (
    compute_class_f1,
    compute_harmonic_mean_f1,
    compute_localization_iou,
    compute_calibration_metrics,
    DAMAGE_CLASS_NAMES
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Disaster Damage Assessment Checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint file to evaluate")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Root directory containing disaster dataset")
    parser.add_argument("--split", type=str, choices=["test", "val", "train"], default="test",
                        help="Dataset partition to evaluate on")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile resolution")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--measure-calibration", action="store_true", default=True,
                        help="Compute ECE, MCE, and Brier calibration metrics")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional file path to persist evaluation results")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Allow overwriting existing evaluation JSON reports")
    parser.add_argument("--splits-file", type=str, default=None,
                        help="Optional explicit path to splits.json file to use for evaluation")
    return parser.parse_args()


def resolve_output_path(
    output_json: Optional[str],
    checkpoint: str,
    split: str,
    model_type: str,
    overwrite: bool = False
) -> str:
    """
    Determines output report path and prevents unintended overwrites.

    Requirements:
      - If explicit output_json already exists and overwrite=False, raises FileExistsError.
      - For the default output path, if it already exists and overwrite=False, appends a UTC timestamp suffix.
      - If overwrite=True, permits overwriting either explicit or default path.
    """
    if output_json:
        if os.path.exists(output_json) and not overwrite:
            raise FileExistsError(
                f"Output file '{output_json}' already exists. Pass --overwrite to permit overwriting."
            )
        return output_json

    ckpt_dir = os.path.dirname(checkpoint) or "."
    default_path = os.path.join(ckpt_dir, f"eval_{split}_{model_type}.json")
    if os.path.exists(default_path) and not overwrite:
        timestamp_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        candidate = os.path.join(ckpt_dir, f"eval_{split}_{model_type}_{timestamp_suffix}.json")
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(ckpt_dir, f"eval_{split}_{model_type}_{timestamp_suffix}_{counter}.json")
            counter += 1
        return candidate

    return default_path


def resolve_splits_path(
    checkpoint: str,
    splits_file: Optional[str] = None
) -> str:
    """
    Determines and verifies the split file path for evaluation.

    Requirements:
      - When splits_file is supplied, uses that exact file.
      - Otherwise, requires 'splits.json' in the checkpoint directory.
      - If the required file is missing, raises FileNotFoundError.
        Does not fall back to data_dir or generate replacement splits.
    """
    if splits_file:
        if not os.path.exists(splits_file):
            raise FileNotFoundError(
                f"Explicitly specified split file does not exist: '{splits_file}'"
            )
        return splits_file

    ckpt_dir = os.path.dirname(checkpoint) or DEFAULT_CHECKPOINT_DIR
    default_splits_file = os.path.join(ckpt_dir, "splits.json")
    if not os.path.exists(default_splits_file):
        raise FileNotFoundError(
            f"Required split file 'splits.json' not found in checkpoint directory: '{ckpt_dir}'. "
            f"Evaluation requires the exact split used during training. Replacement splits are not generated."
        )
    return default_splits_file


def main():
    args = parse_args()

    # Early check for explicit output overwrite protection
    if args.output_json and os.path.exists(args.output_json) and not args.overwrite:
        print(f"[ERROR] Output file '{args.output_json}' already exists. Pass --overwrite to permit overwriting.")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  INDEPENDENT MODEL EVALUATION & CALIBRATION CERTIFICATION")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Split: {args.split.upper()} | Device: {device.upper()}")
    print("=" * 70)

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint file does not exist: {args.checkpoint}")
        sys.exit(1)

    # 1. Scan and split dataset
    samples = scan_xbd_directory(args.data_dir)
    if not samples:
        print(f"[ERROR] No disaster tile pairs found in {args.data_dir}")
        sys.exit(1)

    try:
        splits_path = resolve_splits_path(args.checkpoint, splits_file=args.splits_file)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        splits = create_reproducible_splits(samples, output_dir=splits_path)
    except ValueError as e:
        print(f"[ERROR] Split file validation failed for '{splits_path}': {e}")
        sys.exit(1)
    eval_samples = splits.get(args.split, [])

    if not eval_samples:
        print(f"[ERROR] No samples found in split '{args.split}'")
        sys.exit(1)

    eval_ds = xBDDisasterDataset(eval_samples, tile_size=args.tile_size, augment=False)
    dataloader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Determine model architecture from filename or checkpoint payload
    try:
        ckpt_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt_payload = torch.load(args.checkpoint, map_location=device)
    model_type = "SiameseDamageUNet"
    if isinstance(ckpt_payload, dict) and "model_type" in ckpt_payload:
        model_type = ckpt_payload["model_type"]
    elif "localization" in os.path.basename(args.checkpoint).lower():
        model_type = "BuildingLocalizationUNet"

    print(f"\n[*] Model Architecture: {model_type}")
    print(f"[*] Evaluating on {len(eval_samples)} {args.split} tiles...")

    results = {
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "evaluated_split": args.split,
        "sample_count": len(eval_samples),
        "timestamp": ckpt_payload.get("timestamp") if isinstance(ckpt_payload, dict) else None
    }

    if model_type == "BuildingLocalizationUNet":
        model = BuildingLocalizationUNet(in_channels=3, out_channels=1, base_filters=32).to(device)
        if isinstance(ckpt_payload, dict) and "model_state_dict" in ckpt_payload:
            model.load_state_dict(ckpt_payload["model_state_dict"])
        elif isinstance(ckpt_payload, dict) and "state_dict" in ckpt_payload:
            model.load_state_dict(ckpt_payload["state_dict"])
        else:
            model.load_state_dict(ckpt_payload)

        model.eval()
        iou_scores = []
        with torch.no_grad():
            for batch in dataloader:
                pre = batch["pre_img"].to(device)
                loc_target = batch["loc_mask"].to(device)
                logits = model(pre)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                for p, t in zip(preds, loc_target):
                    iou_scores.append(compute_localization_iou(p, t))

        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        results["mean_iou"] = round(mean_iou, 4)
        print(f"\n[RESULTS] Building Localization Mean IoU: {mean_iou:.4f}")

    else:
        model = SiameseDamageUNet(in_channels=3, num_classes=4, base_filters=32).to(device)
        if isinstance(ckpt_payload, dict) and "model_state_dict" in ckpt_payload:
            model.load_state_dict(ckpt_payload["model_state_dict"])
        elif isinstance(ckpt_payload, dict) and "state_dict" in ckpt_payload:
            model.load_state_dict(ckpt_payload["state_dict"])
        else:
            model.load_state_dict(ckpt_payload)

        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                pre = batch["pre_img"].to(device)
                post = batch["post_img"].to(device)
                target = batch["damage_mask"].to(device)

                logits = model(pre, post)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(target.cpu().numpy())

        preds_arr = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        targets_arr = np.concatenate(all_targets, axis=0) if all_targets else np.array([])

        class_f1_res = compute_class_f1(preds_arr, targets_arr, num_classes=4)
        class_f1s = {c: class_f1_res["classes"][c]["f1"] for c in range(4)}
        harmonic_f1 = compute_harmonic_mean_f1(class_f1s)

        results["harmonic_mean_f1"] = harmonic_f1
        results["macro_f1"] = class_f1_res["macro_f1"]
        results["class_metrics"] = class_f1_res["classes"]
        results["confusion_matrix"] = class_f1_res["confusion_matrix"]

        print("\n" + "-" * 70)
        print("  DAMAGE CLASSIFICATION PERFORMANCE METRICS")
        print("-" * 70)
        print(f"{'Class ID':<10} {'Category':<16} {'Precision':<12} {'Recall':<12} {'F1-Score':<10} {'Support'}")
        for c, m in class_f1_res["classes"].items():
            print(f"{c:<10} {m['name']:<16} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<10.4f} {m['support']}")
        print("-" * 70)
        print(f"  Macro F1:          {class_f1_res['macro_f1']:.4f}")
        print(f"  xView2 Harmonic F1: {harmonic_f1:.4f}")
        print("-" * 70)

        if args.measure_calibration and all_probs:
            probs_arr = np.concatenate(all_probs, axis=0)

            # Detect dataset origin using unified detection logic
            origin_info = detect_dataset_origin(args.data_dir)
            is_synthetic = bool(origin_info.get("is_synthetic", False))
            is_unknown = bool(origin_info.get("is_unknown_origin", False))
            is_op = bool(origin_info.get("is_operational", False))
            denial_reason = origin_info.get("disqualification_reason")

            cal_metrics = compute_calibration_metrics(
                probs_arr,
                targets_arr,
                num_bins=10,
                num_tiles=len(eval_samples),
                min_tiles_for_certification=50,
                is_synthetic_benchmark=is_synthetic,
                is_unknown_origin=is_unknown,
                is_operational=is_op,
                provenance_denial_reason=denial_reason
            )
            results["calibration"] = cal_metrics

            print("\n" + "-" * 70)
            print("  PROBABILITY CALIBRATION DIAGNOSTICS")
            print("-" * 70)
            origin_label = "Synthetic Benchmark" if is_synthetic else ("Operational Imagery (Verified)" if is_op else "Unverified / Unknown Provenance")
            print(f"  Evaluated Tiles:                  {len(eval_samples)} ({origin_label})")
            print(f"  Evaluated Pixels / Footprints:    {cal_metrics['num_samples']}")
            print(f"  Expected Calibration Error (ECE): {cal_metrics['ece']:.4f}")
            print(f"  Maximum Calibration Error (MCE):  {cal_metrics['mce']:.4f}")
            print(f"  Multi-Class Brier Score:          {cal_metrics['brier_score']:.4f}")
            if cal_metrics["is_calibrated"]:
                print("  Certification Status: [CERTIFIED CALIBRATED]")
                print("  ECE < 0.10 threshold satisfied across >= 50 genuinely held-out operational tiles.")
            else:
                print(f"  Certification Status: [{cal_metrics['certification_status']}]")
                for reason in cal_metrics.get("disqualification_reasons", []):
                    print(f"    - {reason}")
            print("-" * 70)

    # Save output report
    try:
        out_file = resolve_output_path(
            output_json=args.output_json,
            checkpoint=args.checkpoint,
            split=args.split,
            model_type=model_type,
            overwrite=args.overwrite
        )
    except FileExistsError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Evaluation report written to: {out_file}")


if __name__ == "__main__":
    main()
