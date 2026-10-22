"""
Evaluation Metrics & Calibration Measurement for Disaster Damage Assessment
Implements:
  - Multi-class Precision, Recall, and F1 Score
  - xView2 Official Harmonic Mean F1 Score across damage categories
  - Localization IoU (Intersection-over-Union)
  - Overall Composite Disaster Damage Score (0.3 * F1_loc + 0.7 * F1_damage)
  - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  - Multi-class Brier Score
"""

import numpy as np
import torch
from typing import Dict, List, Any, Union, Optional, Tuple


DAMAGE_CLASS_NAMES = {
    0: "undamaged",
    1: "minor_damage",
    2: "major_damage",
    3: "destroyed"
}


def _to_numpy(tensor_or_arr: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(tensor_or_arr, torch.Tensor):
        return tensor_or_arr.detach().cpu().numpy()
    return np.asarray(tensor_or_arr)


def compute_confusion_matrix(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: int = 4
) -> np.ndarray:
    """
    Computes (num_classes, num_classes) confusion matrix where:
    rows = true class, columns = predicted class.
    """
    y_pred = _to_numpy(preds).flatten()
    y_true = _to_numpy(targets).flatten()

    valid_mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    y_pred = y_pred[valid_mask]
    y_true = y_true[valid_mask]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if len(y_true) > 0:
        np.add.at(cm, (y_true, y_pred), 1)
    return cm


def compute_class_f1(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: int = 4
) -> Dict[str, Any]:
    """
    Computes per-class Precision, Recall, F1-Score, and Macro F1.
    """
    cm = compute_confusion_matrix(preds, targets, num_classes=num_classes)
    results = {
        "classes": {},
        "macro_f1": 0.0,
        "confusion_matrix": cm.tolist()
    }

    f1_list = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)
        support = int(np.sum(cm[c, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        c_name = DAMAGE_CLASS_NAMES.get(c, f"class_{c}")
        results["classes"][c] = {
            "name": c_name,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support
        }
        f1_list.append(f1)

    results["macro_f1"] = round(float(np.mean(f1_list)), 4) if f1_list else 0.0
    return results


def compute_harmonic_mean_f1(class_f1_scores: Union[List[float], Dict[int, float]], eps: float = 1e-6) -> float:
    """
    Official xView2 damage classification metric:
    Harmonic mean of individual class F1 scores:
      F1_damage = 4 / sum(1 / max(F1_c, eps) for c in [0..3])
    """
    if isinstance(class_f1_scores, dict):
        scores = [float(class_f1_scores[k]) for k in sorted(class_f1_scores.keys())]
    else:
        scores = [float(s) for s in class_f1_scores]

    if len(scores) == 0:
        return 0.0

    denom = sum(1.0 / max(s, eps) for s in scores)
    harmonic_mean = len(scores) / denom
    return round(float(harmonic_mean), 4)


def compute_localization_iou(
    pred_mask: Union[torch.Tensor, np.ndarray],
    target_mask: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> float:
    """
    Computes Intersection-over-Union (Jaccard Index) for binary building footprints.
    Handles binary integer masks and continuous float probability maps with numerical stability.
    """
    p_arr = _to_numpy(pred_mask)
    t_arr = _to_numpy(target_mask)

    if np.issubdtype(p_arr.dtype, np.floating):
        p = (p_arr >= threshold).flatten()
    else:
        p = (p_arr > 0).flatten()

    if np.issubdtype(t_arr.dtype, np.floating):
        t = (t_arr >= threshold).flatten()
    else:
        t = (t_arr > 0).flatten()

    if len(p) == 0 and len(t) == 0:
        return 1.0

    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return round(float(intersection / union), 4)


def compute_composite_score(f1_loc: float, f1_damage: float) -> float:
    """
    Official xView2 composite score:
    Score = 0.3 * F1_loc + 0.7 * F1_damage
    """
    return round(0.3 * float(f1_loc) + 0.7 * float(f1_damage), 4)


def compute_calibration_metrics(
    probs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_bins: int = 10,
    ece_threshold: float = 0.10,
    min_samples_for_certification: int = 50,
    num_tiles: Optional[int] = None,
    min_tiles_for_certification: int = 50,
    is_synthetic_benchmark: bool = False
) -> Dict[str, Any]:
    """
    Calculates calibration metrics for confidence estimation:
      - Expected Calibration Error (ECE) across M confidence bins
      - Maximum Calibration Error (MCE)
      - Multi-class Brier Score: Mean squared error of probabilities vs one-hot targets
      - is_calibrated: True ONLY IF ECE < ece_threshold, sample size is at least 50
        genuinely held-out disaster tiles, and data is NOT from a synthetic benchmark.
    """
    p_arr = _to_numpy(probs)
    t_arr = _to_numpy(targets).astype(np.int64)

    # Flatten spatial dimensions if needed: (N, C)
    if p_arr.ndim == 4:  # (B, C, H, W)
        p_arr = np.transpose(p_arr, (0, 2, 3, 1)).reshape(-1, p_arr.shape[1])
        t_arr = t_arr.flatten()
    elif p_arr.ndim == 3 and p_arr.shape[0] == 4:  # (C, H, W)
        p_arr = np.transpose(p_arr, (1, 2, 0)).reshape(-1, 4)
        t_arr = t_arr.flatten()

    # Filter invalid target pixels if any
    num_classes = p_arr.shape[1] if p_arr.ndim == 2 and p_arr.shape[1] > 0 else 4
    valid = (t_arr >= 0) & (t_arr < num_classes)
    p_arr = p_arr[valid]
    t_arr = t_arr[valid]

    n_samples = len(t_arr)
    if n_samples == 0:
        return {
            "ece": 1.0,
            "mce": 1.0,
            "brier_score": 1.0,
            "is_calibrated": False,
            "certification_status": "UNCERTIFIED_EMPTY_EVALUATION",
            "disqualification_reasons": ["Evaluation set contained 0 valid pixel targets."],
            "num_samples": 0,
            "num_tiles": num_tiles or 0,
            "is_synthetic_benchmark": is_synthetic_benchmark,
            "bins": []
        }

    confidences = np.clip(np.max(p_arr, axis=1), 0.0, 1.0)
    predictions = np.argmax(p_arr, axis=1)
    accuracies = (predictions == t_arr).astype(np.float64)

    # Brier score: (1/N) * sum_i sum_c (p_ic - y_ic)^2
    one_hot = np.zeros_like(p_arr)
    one_hot[np.arange(n_samples), t_arr] = 1.0
    brier_score = float(np.mean(np.sum((p_arr - one_hot) ** 2, axis=1)))

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / n_samples) * bin_error
            mce = max(mce, bin_error)

            bin_details.append({
                "bin_idx": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "calibration_gap": round(bin_error, 4)
            })

    # Strict audit certification validation
    disqualification_reasons = []
    if is_synthetic_benchmark:
        disqualification_reasons.append("Evaluation conducted on synthetic benchmark dataset; operational calibration cannot be certified on synthetic data.")
    if num_tiles is not None and num_tiles < min_tiles_for_certification:
        disqualification_reasons.append(f"Evaluated sample volume ({num_tiles} tiles) is below the minimum operational certification threshold of {min_tiles_for_certification} held-out tiles.")
    if n_samples < min_samples_for_certification:
        disqualification_reasons.append(f"Evaluated pixel count ({n_samples}) is below minimum sample threshold ({min_samples_for_certification}).")
    if ece >= ece_threshold:
        disqualification_reasons.append(f"Expected Calibration Error ({ece:.4f}) exceeds the {ece_threshold:.2f} certification threshold.")

    if len(disqualification_reasons) == 0:
        is_calibrated = True
        certification_status = "CERTIFIED_CALIBRATED"
    else:
        is_calibrated = False
        certification_status = "UNCERTIFIED_CALIBRATION"

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(brier_score, 4),
        "is_calibrated": is_calibrated,
        "certification_status": certification_status,
        "disqualification_reasons": disqualification_reasons,
        "num_samples": n_samples,
        "num_tiles": num_tiles or 0,
        "is_synthetic_benchmark": is_synthetic_benchmark,
        "bins": bin_details
    }

