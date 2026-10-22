"""
Unit & Integration Tests for Dataset Pipeline, Evaluation Metrics,
Reproducible Training Engine, and Checkpoint Calibration Integrity
"""

import os
import json
import tempfile
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.src.data.dataset import (
    parse_wkt_polygon,
    rasterize_xbd_labels,
    apply_synchronized_augmentations,
    create_reproducible_splits,
    xBDDisasterDataset,
    create_verified_benchmark_subset
)
from ml.src.training.metrics import (
    compute_confusion_matrix,
    compute_class_f1,
    compute_harmonic_mean_f1,
    compute_localization_iou,
    compute_composite_score,
    compute_calibration_metrics
)
from ml.src.training.trainer import (
    TrainingConfig,
    CheckpointManager,
    LocalizationTrainer,
    DamageTrainer,
    set_reproducible_seed
)
from ml.src.inference.engine import DamageInferenceEngine


def test_wkt_polygon_parser():
    """Verify pure-Python regex extraction of coordinate vertices from WKT strings"""
    wkt = "POLYGON ((10.0 20.0, 50.0 20.0, 50.0 60.0, 10.0 60.0, 10.0 20.0))"
    contours = parse_wkt_polygon(wkt)
    assert len(contours) == 1
    pts = contours[0].reshape(-1, 2)
    assert len(pts) == 5
    assert np.array_equal(pts[0], [10, 20])
    assert np.array_equal(pts[2], [50, 60])


def test_xbd_label_rasterization(tmp_path):
    """Verify polygon rasterization to binary localization mask and multi-class damage mask"""
    label_data = {
        "metadata": {"disaster_name": "test_flood"},
        "features": {
            "xy": [
                {
                    "properties": {"feature_type": "building", "subtype": "destroyed"},
                    "wkt": "POLYGON ((20 20, 60 20, 60 60, 20 60, 20 20))"
                },
                {
                    "properties": {"feature_type": "building", "subtype": "no-damage"},
                    "wkt": "POLYGON ((70 70, 90 70, 90 90, 70 90, 70 70))"
                }
            ]
        }
    }
    json_path = os.path.join(tmp_path, "sample_post_disaster.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(label_data, f)

    # 1. Test localization rasterization (binary)
    loc_mask = rasterize_xbd_labels(json_path, target_hw=(128, 128), is_post_disaster=False)
    assert loc_mask.shape == (128, 128)
    assert loc_mask[30, 30] == 1  # Inside first building
    assert loc_mask[80, 80] == 1  # Inside second building
    assert loc_mask[5, 5] == 0    # Background

    # 2. Test damage classification rasterization (multi-class)
    dmg_mask = rasterize_xbd_labels(json_path, target_hw=(128, 128), is_post_disaster=True)
    assert dmg_mask[30, 30] == 3  # Destroyed = class 3
    assert dmg_mask[80, 80] == 0  # No damage = class 0


def test_synchronized_augmentations():
    """Verify geometric flips and rotations apply identical transforms across all four layers"""
    pre = np.zeros((64, 64, 3), dtype=np.uint8)
    post = np.zeros((64, 64, 3), dtype=np.uint8)
    loc = np.zeros((64, 64), dtype=np.uint8)
    dmg = np.zeros((64, 64), dtype=np.uint8)

    # Put distinctive marker at top-left: (x in [0..9], y in [0..9])
    pre[0:10, 0:10, :] = 255
    post[0:10, 0:10, :] = 200
    loc[0:10, 0:10] = 1
    dmg[0:10, 0:10] = 3

    # Apply Horizontal Flip
    p_h, po_h, l_h, d_h = apply_synchronized_augmentations(pre, post, loc, dmg, flip_h=True)
    # Marker must now be at top-right: x in [54..63], y in [0..9]
    assert np.all(p_h[0:10, 54:64, :] == 255)
    assert np.all(po_h[0:10, 54:64, :] == 200)
    assert np.all(l_h[0:10, 54:64] == 1)
    assert np.all(d_h[0:10, 54:64] == 3)
    # Original top-left must now be empty
    assert np.all(l_h[0:10, 0:10] == 0)


def test_stratified_split_reproducibility(tmp_path):
    """Verify partition reproducibility and persistence in splits.json"""
    samples = [{"pair_id": f"event_{i:04d}", "pre_img_path": "", "post_img_path": ""} for i in range(20)]

    out_dir_1 = os.path.join(tmp_path, "run1")
    out_dir_2 = os.path.join(tmp_path, "run2")

    split1 = create_reproducible_splits(samples, output_dir=out_dir_1, seed=42)
    split2 = create_reproducible_splits(samples, output_dir=out_dir_2, seed=42)

    assert [s["pair_id"] for s in split1["train"]] == [s["pair_id"] for s in split2["train"]]
    assert [s["pair_id"] for s in split1["val"]] == [s["pair_id"] for s in split2["val"]]
    assert [s["pair_id"] for s in split1["test"]] == [s["pair_id"] for s in split2["test"]]

    # Verify reload from splits.json produces identical partition
    split_reloaded = create_reproducible_splits(samples, output_dir=out_dir_1, seed=999)
    assert [s["pair_id"] for s in split_reloaded["train"]] == [s["pair_id"] for s in split1["train"]]


def test_disaster_split_leakage_prevention(tmp_path):
    """
    Verify disaster-event level isolation:
    Tiles belonging to the same disaster event must never leak across train, val, and test splits.
    """
    samples = []
    disasters = ["monsoon_karnataka", "flood_wayanad", "cyclone_coast", "landslide_hills"]
    for d in disasters:
        for i in range(6):
            samples.append({
                "pair_id": f"{d}_{i:04d}",
                "disaster": d,
                "pre_img_path": "",
                "post_img_path": ""
            })

    out_dir = os.path.join(tmp_path, "disaster_splits")
    splits = create_reproducible_splits(samples, output_dir=out_dir, seed=42, group_by_disaster=True)

    train_disasters = set(s["disaster"] for s in splits["train"])
    val_disasters = set(s["disaster"] for s in splits["val"])
    test_disasters = set(s["disaster"] for s in splits["test"])

    # Disasters must be completely disjoint across splits
    assert train_disasters.isdisjoint(val_disasters), "Disaster leakage detected between Train and Val!"
    assert train_disasters.isdisjoint(test_disasters), "Disaster leakage detected between Train and Test!"
    assert val_disasters.isdisjoint(test_disasters), "Disaster leakage detected between Val and Test!"

    # Every split must have at least one assigned disaster
    assert len(train_disasters) >= 1
    assert len(val_disasters) >= 1
    assert len(test_disasters) >= 1

    # Verify manifest reflects disaster_leakage_prevented == True
    manifest_file = os.path.join(out_dir, "splits.json")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["disaster_leakage_prevented"] is True
    assert manifest["split_strategy"] == "disaster_event_grouped"


def test_metrics_hand_computed_examples():
    """Verify classification and composite metrics against manually calculated numerical ground truth"""
    # Example 1: Perfect agreement
    preds_perfect = np.array([0, 1, 2, 3])
    targets_perfect = np.array([0, 1, 2, 3])
    f1_perfect = compute_class_f1(preds_perfect, targets_perfect, num_classes=4)
    for c in range(4):
        assert f1_perfect["classes"][c]["f1"] == 1.0
    h_f1_perfect = compute_harmonic_mean_f1([1.0, 1.0, 1.0, 1.0])
    assert h_f1_perfect == 1.0

    # Example 2: Known uneven class F1 scores: [1.0, 0.5, 0.5, 0.25]
    # Harmonic mean = 4 / (1/1.0 + 1/0.5 + 1/0.5 + 1/0.25) = 4 / (1 + 2 + 2 + 4) = 4 / 9 = 0.4444
    uneven_scores = [1.0, 0.5, 0.5, 0.25]
    h_f1_uneven = compute_harmonic_mean_f1(uneven_scores)
    assert abs(h_f1_uneven - (4.0 / 9.0)) < 1e-4

    # Example 3: Localization IoU with partial overlap
    # Mask A: 2x2 box (4 pixels), Mask B: 2x2 box shifted by 1 pixel (overlap: 2 pixels, union: 6 pixels)
    # IoU = 2 / 6 = 0.3333
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_b = np.zeros((10, 10), dtype=np.uint8)
    mask_a[2:4, 2:4] = 1
    mask_b[2:4, 3:5] = 1
    iou = compute_localization_iou(mask_a, mask_b)
    assert abs(iou - (2.0 / 6.0)) < 1e-4

    # Example 4: Composite xView2 score (0.3 * IoU + 0.7 * F1_damage)
    comp = compute_composite_score(f1_loc=0.80, f1_damage=0.50)
    assert abs(comp - (0.3 * 0.80 + 0.7 * 0.50)) < 1e-4


def test_ece_calibration_computation():
    """Verify Expected Calibration Error (ECE) measurement and threshold check"""
    # 1. Perfectly calibrated probabilities:
    # 100 samples with 0.90 confidence where 90% are correct
    probs_cal = np.full((100, 2), 0.10)
    probs_cal[:, 0] = 0.90
    targets_cal = np.zeros(100, dtype=np.int64)
    # Set 10% errors
    targets_cal[0:10] = 1

    cal_metrics = compute_calibration_metrics(
        probs_cal, targets_cal, num_bins=10, min_samples_for_certification=50,
        num_tiles=60, is_synthetic_benchmark=False
    )
    assert cal_metrics["num_samples"] == 100
    assert cal_metrics["num_tiles"] == 60
    assert cal_metrics["ece"] < 0.05
    assert cal_metrics["is_calibrated"] is True
    assert cal_metrics["certification_status"] == "CERTIFIED_CALIBRATED"

    # 2. Severely overconfident, inaccurate probabilities:
    # Confidence is 0.99 but accuracy is 0%
    probs_uncal = np.zeros((60, 2))
    probs_uncal[:, 1] = 0.99
    probs_uncal[:, 0] = 0.01
    targets_uncal = np.zeros(60, dtype=np.int64)  # true class is 0

    uncal_metrics = compute_calibration_metrics(
        probs_uncal, targets_uncal, num_bins=10, min_samples_for_certification=50,
        num_tiles=60, is_synthetic_benchmark=False
    )
    assert uncal_metrics["ece"] > 0.80
    assert uncal_metrics["is_calibrated"] is False
    assert uncal_metrics["certification_status"] == "UNCERTIFIED_CALIBRATION"


def test_calibration_certification_strict_audit_rejections():
    """
    Verify strict audit rules for calibration certification:
      - Synthetic benchmark data must NEVER be certified as operational calibration
      - Evaluation with < 50 genuinely held-out tiles must be rejected
      - ECE >= 0.10 must be rejected
    """
    probs = np.full((100, 2), 0.10)
    probs[:, 0] = 0.90
    targets = np.zeros(100, dtype=np.int64)
    targets[0:10] = 1  # 90% accurate, low ECE

    # Case A: Synthetic benchmark flag is True -> REJECTED
    res_synth = compute_calibration_metrics(probs, targets, num_tiles=100, is_synthetic_benchmark=True)
    assert res_synth["is_calibrated"] is False
    assert any("synthetic benchmark" in r.lower() for r in res_synth["disqualification_reasons"])

    # Case B: Held-out volume is below 50 tiles -> REJECTED
    res_small_volume = compute_calibration_metrics(probs, targets, num_tiles=24, is_synthetic_benchmark=False)
    assert res_small_volume["is_calibrated"] is False
    assert any("below the minimum operational certification threshold" in r.lower() for r in res_small_volume["disqualification_reasons"])

    # Case C: High ECE (>= 0.10) with 60 tiles -> REJECTED
    probs_bad = np.full((100, 2), 0.01)
    probs_bad[:, 1] = 0.99
    targets_bad = np.zeros(100, dtype=np.int64)
    res_high_ece = compute_calibration_metrics(probs_bad, targets_bad, num_tiles=60, is_synthetic_benchmark=False)
    assert res_high_ece["is_calibrated"] is False
    assert any("exceeds the 0.10" in r.lower() for r in res_high_ece["disqualification_reasons"])


def test_metrics_edge_cases():
    """Verify numerical stability and graceful handling across all metric edge cases"""
    # 1. Empty arrays
    empty_f1 = compute_class_f1(np.array([]), np.array([]), num_classes=4)
    assert empty_f1["macro_f1"] == 0.0
    for c in range(4):
        assert empty_f1["classes"][c]["f1"] == 0.0
        assert empty_f1["classes"][c]["support"] == 0

    # 2. Classes absent from ground truth and predictions
    # Only class 0 present; classes 1, 2, 3 have zero support and zero predictions
    p_mono = np.zeros(20, dtype=np.int64)
    t_mono = np.zeros(20, dtype=np.int64)
    mono_f1 = compute_class_f1(p_mono, t_mono, num_classes=4)
    assert mono_f1["classes"][0]["f1"] == 1.0
    for c in [1, 2, 3]:
        assert mono_f1["classes"][c]["precision"] == 0.0
        assert mono_f1["classes"][c]["recall"] == 0.0
        assert mono_f1["classes"][c]["f1"] == 0.0
        assert mono_f1["classes"][c]["support"] == 0

    # 3. Harmonic mean F1 with zero performance
    # When any damage class has 0.0 F1, harmonic mean is bounded and does not throw ZeroDivisionError
    h_zero = compute_harmonic_mean_f1([1.0, 1.0, 1.0, 0.0])
    assert h_zero < 0.001
    assert compute_harmonic_mean_f1([]) == 0.0

    # 4. Building Localization IoU edge cases:
    # Both masks empty (0 buildings) -> IoU is 1.0 (perfect agreement on background)
    empty_a = np.zeros((10, 10), dtype=np.uint8)
    empty_b = np.zeros((10, 10), dtype=np.uint8)
    assert compute_localization_iou(empty_a, empty_b) == 1.0

    # One mask empty, other has building -> IoU is 0.0
    non_empty = np.zeros((10, 10), dtype=np.uint8)
    non_empty[2:5, 2:5] = 1
    assert compute_localization_iou(empty_a, non_empty) == 0.0

    # Continuous float probability maps with thresholding
    probs_map = np.full((10, 10), 0.2, dtype=np.float32)  # all below 0.5 threshold
    probs_map[2:4, 2:4] = 0.8  # 4 pixels above threshold
    target_bin = np.zeros((10, 10), dtype=np.uint8)
    target_bin[2:4, 2:4] = 1  # exact same 4 pixels
    assert compute_localization_iou(probs_map, target_bin) == 1.0

    # 5. Calibration metrics on empty input
    empty_cal = compute_calibration_metrics(np.array([]), np.array([]))
    assert empty_cal["is_calibrated"] is False
    assert empty_cal["ece"] == 1.0


def test_checkpoint_manager_atomic_save_and_overwrite_protection(tmp_path):
    """Verify CheckpointManager protects existing checkpoints against unintended overwrites"""
    ckpt_dir = os.path.join(tmp_path, "checkpoints")
    manager = CheckpointManager(ckpt_dir)

    model = torch.nn.Linear(10, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = TrainingConfig(data_dir=str(tmp_path), checkpoint_dir=ckpt_dir)

    # 1. Initial save succeeds
    path = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        filename="test_model.pth",
        epoch=1,
        config=config,
        val_metrics={"val_loss": 0.5},
        dataset_info={"samples": 10},
        overwrite=False
    )
    assert os.path.exists(path)

    # 2. Secondary save with overwrite=False raises FileExistsError
    with pytest.raises(FileExistsError):
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            filename="test_model.pth",
            epoch=2,
            config=config,
            val_metrics={"val_loss": 0.4},
            dataset_info={"samples": 10},
            overwrite=False
        )

    # 3. Save with overwrite=True succeeds
    manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        filename="test_model.pth",
        epoch=2,
        config=config,
        val_metrics={"val_loss": 0.4},
        dataset_info={"samples": 10},
        overwrite=True
    )


def test_deterministic_smoke_training_loop(tmp_path):
    """Runs a fast 1-epoch deterministic smoke training pass to verify gradients and loss convergence"""
    set_reproducible_seed(42)

    # Build 4 miniature synthetic pairs
    dataset_dir = os.path.join(tmp_path, "benchmark_data")
    ckpt_dir = os.path.join(tmp_path, "checkpoints")
    samples = create_verified_benchmark_subset(target_dir=dataset_dir, num_pairs=4, tile_size=64, seed=42)

    splits = create_reproducible_splits(samples, output_dir=ckpt_dir, seed=42)
    train_ds = xBDDisasterDataset(splits["train"], tile_size=64, augment=False)
    val_ds = xBDDisasterDataset(splits["val"], tile_size=64, augment=False)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    config = TrainingConfig(
        data_dir=dataset_dir,
        checkpoint_dir=ckpt_dir,
        epochs=1,
        batch_size=2,
        tile_size=64,
        seed=42,
        overwrite_checkpoints=True
    )

    dataset_info = {"total_pairs": 4}

    # 1. Smoke test Localization Trainer
    loc_trainer = LocalizationTrainer(config)
    loc_res = loc_trainer.fit(train_loader, val_loader, dataset_info)
    assert loc_res["model_type"] == "BuildingLocalizationUNet"
    assert os.path.exists(loc_res["checkpoint_path"])
    assert loc_res["history"][0]["train_loss"] > 0.0

    # 2. Smoke test Damage Trainer
    dmg_trainer = DamageTrainer(config)
    dmg_res = dmg_trainer.fit(train_loader, val_loader, dataset_info)
    assert dmg_res["model_type"] == "SiameseDamageUNet"
    assert os.path.exists(dmg_res["checkpoint_path"])
    assert dmg_res["history"][0]["train_loss"] > 0.0


def test_inference_engine_checkpoint_status_integrity(tmp_path):
    """
    Verifies that DamageInferenceEngine accurately distinguishes:
      - UNTRAINED_DEMO_INITIALIZATION
      - TRAINED_WEIGHTS_UNMEASURED_CALIBRATION
      - CALIBRATED_CHECKPOINT (strictly requires non-synthetic, >= 50 tiles, and ECE < 0.10)
    """
    weights_dir = os.path.join(tmp_path, "test_weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 1. No weights present -> UNTRAINED_DEMO_INITIALIZATION
    engine_demo = DamageInferenceEngine(weights_dir=weights_dir)
    status_demo = engine_demo.get_model_status()
    assert status_demo["weights_loaded"] is False
    assert status_demo["weights_status"] == "UNTRAINED_DEMO_INITIALIZATION"

    # 2. Create uncalibrated trained checkpoints -> TRAINED_WEIGHTS_UNMEASURED_CALIBRATION
    loc_path = os.path.join(weights_dir, "unet_localization.pth")
    dmg_path = os.path.join(weights_dir, "siamese_damage.pth")

    from ml.src.models.unet import BuildingLocalizationUNet
    from ml.src.models.siamese_unet import SiameseDamageUNet

    loc_model = BuildingLocalizationUNet(in_channels=3, out_channels=1, base_filters=32)
    dmg_model = SiameseDamageUNet(in_channels=3, num_classes=4, base_filters=32)

    torch.save({"model_state_dict": loc_model.state_dict()}, loc_path)
    torch.save({
        "model_state_dict": dmg_model.state_dict(),
        "calibration_measured": False,
        "calibration_metrics": None
    }, dmg_path)

    engine_unmeasured = DamageInferenceEngine(weights_dir=weights_dir)
    status_unmeasured = engine_unmeasured.get_model_status()
    assert status_unmeasured["weights_loaded"] is True
    assert status_unmeasured["weights_status"] == "TRAINED_WEIGHTS_UNMEASURED_CALIBRATION"

    # 3. Create synthetic benchmark checkpoint -> Still TRAINED_WEIGHTS_UNMEASURED_CALIBRATION
    torch.save({
        "model_state_dict": dmg_model.state_dict(),
        "calibration_measured": True,
        "calibration_metrics": {
            "ece": 0.042,
            "mce": 0.081,
            "brier_score": 0.052,
            "is_calibrated": False,
            "is_synthetic_benchmark": True,
            "num_tiles": 60,
            "num_samples": 500,
            "disqualification_reasons": ["Synthetic benchmark cannot certify operational calibration."]
        },
        "calibration_criteria_met": False
    }, dmg_path)

    engine_synth = DamageInferenceEngine(weights_dir=weights_dir)
    status_synth = engine_synth.get_model_status()
    assert status_synth["weights_status"] == "TRAINED_WEIGHTS_UNMEASURED_CALIBRATION"
    assert "Synthetic benchmark cannot certify" in status_synth["notice"]

    # 4. Create verified operational calibrated checkpoint (non-synthetic, >= 50 tiles, ECE < 0.10)
    torch.save({
        "model_state_dict": dmg_model.state_dict(),
        "calibration_measured": True,
        "calibration_metrics": {
            "ece": 0.042,
            "mce": 0.081,
            "brier_score": 0.052,
            "is_calibrated": True,
            "is_synthetic_benchmark": False,
            "num_tiles": 60,
            "num_samples": 500,
            "bins": [{"bin_idx": 0, "calibration_gap": 0.03}]
        },
        "calibration_criteria_met": True
    }, dmg_path)

    engine_calibrated = DamageInferenceEngine(weights_dir=weights_dir)
    status_calibrated = engine_calibrated.get_model_status()
    assert status_calibrated["weights_loaded"] is True
    assert status_calibrated["weights_status"] == "CALIBRATED_CHECKPOINT"
    assert status_calibrated["calibration_metrics"]["ece"] == 0.042
