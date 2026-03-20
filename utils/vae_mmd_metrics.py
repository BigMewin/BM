#!/usr/bin/env python3
"""
Segmentation evaluation metrics for brain metastasis experiments.

Includes:
    - compute_surface_distances: surface-to-surface distance arrays
    - compute_hd95: 95th percentile Hausdorff Distance
    - compute_surface_dice: Surface Dice (sDice) with configurable tolerance
    - compute_metrics: all metrics for a single prediction/GT pair
    - avg_metrics: aggregate metrics across a list of per-case results
    - print_metrics: formatted console output
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Surface distance primitives
# ---------------------------------------------------------------------------

def compute_surface_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute surface-to-surface distance arrays between prediction and GT.

    Args:
        pred: Binary prediction volume, shape (D, H, W).
        gt: Binary ground-truth volume, shape (D, H, W).
        spacing: Voxel spacing in mm (z, y, x).

    Returns:
        dist_pred_to_gt: Distances from each pred surface voxel to GT surface.
        dist_gt_to_pred: Distances from each GT surface voxel to pred surface.
        Either array is None if no surface voxels exist.
    """
    struct = generate_binary_structure(3, 1)
    pred_border = pred ^ binary_erosion(pred, structure=struct)
    gt_border = gt ^ binary_erosion(gt, structure=struct)

    if not np.any(pred_border) or not np.any(gt_border):
        return None, None

    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)

    return dt_gt[pred_border], dt_pred[gt_border]


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    95th percentile Hausdorff Distance in mm.

    Returns np.nan if either volume is empty or has no surface.

    Args:
        pred: Binary prediction volume.
        gt: Binary ground-truth volume.
        spacing: Voxel spacing in mm.

    Returns:
        HD95 value in mm, or np.nan.
    """
    pred_binary = (pred > 0).astype(bool)
    gt_binary = (gt > 0).astype(bool)

    if not np.any(pred_binary) or not np.any(gt_binary):
        return np.nan

    d_p2g, d_g2p = compute_surface_distances(pred_binary, gt_binary, spacing)
    if d_p2g is None or d_g2p is None:
        return np.nan

    return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))


def compute_surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: float = 1.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Surface Dice (sDice) with configurable tolerance.

    sDice = (|S_pred within tol of S_gt| + |S_gt within tol of S_pred|)
            / (|S_pred| + |S_gt|)

    Args:
        pred: Binary prediction volume.
        gt: Binary ground-truth volume.
        tolerance: Surface tolerance distance in mm.
        spacing: Voxel spacing in mm.

    Returns:
        sDice in [0, 1]. Returns 1.0 if both volumes are empty.
    """
    pred_binary = (pred > 0).astype(bool)
    gt_binary = (gt > 0).astype(bool)

    if not np.any(pred_binary) and not np.any(gt_binary):
        return 1.0
    if not np.any(pred_binary) or not np.any(gt_binary):
        return 0.0

    d_p2g, d_g2p = compute_surface_distances(pred_binary, gt_binary, spacing)
    if d_p2g is None or d_g2p is None:
        return 0.0

    n_pred_within = np.sum(d_p2g <= tolerance)
    n_gt_within = np.sum(d_g2p <= tolerance)

    return float((n_pred_within + n_gt_within) / (len(d_p2g) + len(d_g2p)))


# ---------------------------------------------------------------------------
# Composite metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sdice_tolerance: float = 1.0,
) -> dict:
    """
    Compute all segmentation metrics for a single prediction/GT pair.

    Metrics:
        sensitivity: TP / (TP + FN)
        precision:   TP / (TP + FP)
        f1:          2*TP / (2*TP + FP + FN)
        f2:          5*TP / (5*TP + FP + 4*FN)
        sdice:       Surface Dice at given tolerance
        hd95:        95th percentile Hausdorff Distance in mm

    Args:
        pred: Prediction volume (any dtype; thresholded at 0).
        gt: Ground-truth volume (any dtype; thresholded at 0).
        spacing: Voxel spacing in mm for surface metrics.
        sdice_tolerance: Tolerance in mm for sDice computation.

    Returns:
        Dictionary with keys: sensitivity, precision, f1, f2, sdice, hd95.
    """
    pred_flat = (pred > 0).astype(np.int32).flatten()
    gt_flat = (gt > 0).astype(np.int32).flatten()

    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    f2 = 5 * tp / (5 * tp + fp + 4 * fn) if (5 * tp + fp + 4 * fn) > 0 else 0.0

    sdice = compute_surface_dice(pred, gt, tolerance=sdice_tolerance, spacing=spacing)
    hd95 = compute_hd95(pred, gt, spacing=spacing)

    return {
        "sensitivity": sensitivity,
        "precision":   precision,
        "f1":          f1,
        "f2":          f2,
        "sdice":       sdice,
        "hd95":        hd95,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def avg_metrics(results_list: list) -> dict:
    """
    Aggregate per-case metrics across a list of result dicts.

    sensitivity, precision, f1, f2, sdice: mean.
    hd95: median (NaN values excluded).

    Args:
        results_list: List of dicts from compute_metrics().

    Returns:
        Dict with same keys as compute_metrics output.
    """
    keys = ["sensitivity", "precision", "f1", "f2", "sdice", "hd95"]

    if not results_list:
        return {k: 0.0 for k in keys}

    avg = {}
    for k in ["sensitivity", "precision", "f1", "f2", "sdice"]:
        avg[k] = float(np.mean([r[k] for r in results_list]))

    hd95_vals = [r["hd95"] for r in results_list if not np.isnan(r["hd95"])]
    avg["hd95"] = float(np.median(hd95_vals)) if hd95_vals else float("nan")

    return avg


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

def print_metrics(dataset_name: str, metrics: dict, n_cases: int, n_total: int):
    """
    Print a formatted metrics summary for one dataset.

    Args:
        dataset_name: Display name (e.g. "Stanford").
        metrics: Output dict from avg_metrics().
        n_cases: Number of successfully evaluated cases.
        n_total: Total number of cases attempted.
    """
    hd95_str = (
        f"{metrics['hd95']:.2f} mm"
        if not np.isnan(metrics["hd95"])
        else "N/A"
    )
    print(f"\n{dataset_name} ({n_cases}/{n_total} cases evaluated)")
    print(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  F2          : {metrics['f2']:.4f}")
    print(f"  sDice       : {metrics['sdice']:.4f}")
    print(f"  HD95        : {hd95_str}")
