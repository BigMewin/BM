#!/usr/bin/env python3
"""
Evaluate a trained nnUNet model on the held-out test set.

Runs inference on the original (non-VAE-reconstructed) test images using
nnUNetPredictor, then computes Sensitivity, Precision, F1, F2, sDice, and
HD95 per case. Results are broken down by dataset and saved as a pickle file.

The test set is always evaluated against the original raw NIfTI files to
preserve evaluation integrity.

Usage:
    python evaluate_nnunet.py \
        --split_file /path/to/four_datasets_split.npz \
        --nnunet_results /path/to/nnUNet_results \
        --dataset_id 998 \
        --dataset_name Dataset998_FourDatasets_VAE \
        --output_dir /path/to/eval_output \
        --stanford_dir /path/to/brainmetshare-3/train \
        --ucsf_dir /path/to/UCSF_BrainMetastases_TRAIN \
        --uclm_dir /path/to/UCLM_data \
        --pkg_dir /path/to/Brain-Mets-Lung-MRI-Path-Segs \
        [--folds 0 1 2 3 4] \
        [--checkpoint checkpoint_best.pth]
"""

import argparse
import logging
import os
import pickle
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import affine_transform
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vae_mmd_metrics import avg_metrics, compute_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nnUNet on the held-out test set."
    )
    parser.add_argument("--split_file",     type=str, required=True)
    parser.add_argument("--nnunet_results", type=str, required=True)
    parser.add_argument("--dataset_id",     type=int, required=True)
    parser.add_argument("--dataset_name",   type=str, required=True)
    parser.add_argument("--output_dir",     type=str, required=True)
    parser.add_argument("--stanford_dir",   type=str, required=True)
    parser.add_argument("--ucsf_dir",       type=str, required=True)
    parser.add_argument("--uclm_dir",       type=str, required=True)
    parser.add_argument("--pkg_dir",        type=str, required=True)
    parser.add_argument("--folds",          type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoint",     type=str, default="checkpoint_best.pth")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-dataset image/label resolvers
# ---------------------------------------------------------------------------

def resolve_stanford(case_full: str, stanford_dir: str):
    case_id = case_full.replace("stanford_", "")
    img = os.path.join(stanford_dir, case_id, "t1_gd.nii.gz")
    seg = os.path.join(stanford_dir, case_id, "seg.nii.gz")
    return img, seg


def resolve_ucsf(case_full: str, ucsf_dir: str):
    case_id = case_full.replace("ucsf_", "")
    img = os.path.join(ucsf_dir, case_id, f"{case_id}_T1post.nii.gz")
    seg = os.path.join(ucsf_dir, case_id, f"{case_id}_seg.nii.gz")
    return img, seg


def resolve_uclm(case_full: str, uclm_dir: str):
    patient = case_full.replace("uclm_", "").rsplit("_", 1)[0]
    patient_path = os.path.join(uclm_dir, patient)
    if not os.path.isdir(patient_path):
        return None, None
    files = os.listdir(patient_path)
    imgs  = sorted([f for f in files if f.endswith(".nii") and "img" in f.lower()])
    masks = sorted([f for f in files if f.endswith(".nii") and "msk" in f.lower()])
    if not imgs or not masks:
        return None, None
    return (
        os.path.join(patient_path, imgs[0]),
        os.path.join(patient_path, masks[0]),
    )


def resolve_pkg(case_full: str, pkg_dir: str):
    folder = case_full.replace("pkg_", "")
    folder_path = os.path.join(pkg_dir, folder)
    if not os.path.isdir(folder_path):
        return None, None
    files = os.listdir(folder_path)
    t1ce  = [f for f in files if "t1ce_img" in f]
    segs  = [f for f in files if "core_seg" in f]
    if not t1ce or not segs:
        return None, None
    return (
        os.path.join(folder_path, t1ce[0]),
        os.path.join(folder_path, segs[0]),
    )


def load_gt(img_path: str, seg_path: str) -> tuple:
    """
    Load GT segmentation aligned to image space.

    For PKG/UCLM where geometry may differ, the segmentation is resampled
    to match the image affine using nearest-neighbour interpolation.

    Returns:
        gt (np.ndarray): Binary segmentation array aligned to image space.
        spacing (tuple): Voxel spacing in mm from the image header.
    """
    img_nib = nib.load(img_path)
    seg_nib = nib.load(seg_path)
    seg_data = seg_nib.get_fdata()

    spacing = tuple(float(v) for v in img_nib.header.get_zooms()[:3])

    needs_alignment = (
        img_nib.shape != seg_nib.shape
        or not np.allclose(img_nib.affine, seg_nib.affine, atol=1e-3)
    )

    if needs_alignment:
        inv_aff = np.linalg.inv(img_nib.affine)
        transform = inv_aff @ seg_nib.affine
        seg_data = affine_transform(
            seg_data,
            matrix=transform[:3, :3],
            offset=transform[:3, 3],
            output_shape=img_nib.shape,
            order=0,
            mode="constant",
            cval=0,
        )

    return (seg_data > 0).astype(np.uint8), spacing


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference_for_domain(
    predictor,
    cases: list,
    resolve_fn,
    domain_args: tuple,
    temp_input: str,
    temp_output: str,
) -> list:
    """
    Run nnUNet inference and compute metrics for all cases in one domain.

    Args:
        predictor: Initialized nnUNetPredictor.
        cases: List of case name stems for this domain.
        resolve_fn: Function(case_full, *domain_args) -> (img_path, seg_path).
        domain_args: Extra arguments forwarded to resolve_fn.
        temp_input: Temporary directory for input files.
        temp_output: Temporary directory for prediction outputs.

    Returns:
        List of per-case metric dicts (keys from compute_metrics + 'case').
    """
    results = []

    for case_full in tqdm(cases, desc=resolve_fn.__name__.replace("resolve_", "").upper()):
        img_path, seg_path = resolve_fn(case_full, *domain_args)

        if img_path is None or not os.path.exists(img_path) or not os.path.exists(seg_path):
            logger.warning("Skipping %s: missing files", case_full)
            continue

        temp_in = os.path.join(temp_input, f"{case_full}_0000.nii.gz")
        shutil.copy(img_path, temp_in)

        try:
            predictor.predict_from_files(
                [[temp_in]], temp_output,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

            pred_path = os.path.join(temp_output, f"{case_full}.nii.gz")
            if not os.path.exists(pred_path):
                logger.warning("No prediction output for %s", case_full)
                continue

            pred = nib.load(pred_path).get_fdata()
            gt, spacing = load_gt(img_path, seg_path)

            m = compute_metrics(pred, gt, spacing=spacing)
            m["case"] = case_full
            results.append(m)

            os.remove(pred_path)

        except Exception as exc:
            logger.error("Inference failed for %s: %s", case_full, exc)
        finally:
            if os.path.exists(temp_in):
                os.remove(temp_in)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    model_folder = os.path.join(
        args.nnunet_results,
        args.dataset_name,
        "nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=tuple(args.folds),
        checkpoint_name=args.checkpoint,
    )
    logger.info("Predictor initialized from %s (folds %s)", model_folder, args.folds)

    split = np.load(args.split_file, allow_pickle=True)
    test_cases = list(split["test_cases"])
    logger.info("Test set: %d cases", len(test_cases))

    stanford_test = [c for c in test_cases if c.startswith("stanford_")]
    ucsf_test     = [c for c in test_cases if c.startswith("ucsf_")]
    uclm_test     = [c for c in test_cases if c.startswith("uclm_")]
    pkg_test      = [c for c in test_cases if c.startswith("pkg_")]

    temp_input  = tempfile.mkdtemp()
    temp_output = tempfile.mkdtemp()

    try:
        stanford_results = run_inference_for_domain(
            predictor, stanford_test, resolve_stanford,
            (args.stanford_dir,), temp_input, temp_output,
        )
        ucsf_results = run_inference_for_domain(
            predictor, ucsf_test, resolve_ucsf,
            (args.ucsf_dir,), temp_input, temp_output,
        )
        uclm_results = run_inference_for_domain(
            predictor, uclm_test, resolve_uclm,
            (args.uclm_dir,), temp_input, temp_output,
        )
        pkg_results = run_inference_for_domain(
            predictor, pkg_test, resolve_pkg,
            (args.pkg_dir,), temp_input, temp_output,
        )
    finally:
        shutil.rmtree(temp_input,  ignore_errors=True)
        shutil.rmtree(temp_output, ignore_errors=True)

    all_results = stanford_results + ucsf_results + uclm_results + pkg_results
    overall = avg_metrics(all_results)

    print("\n" + "=" * 72)
    print("Evaluation Results")
    print("=" * 72)
    print_metrics("Overall",  overall,              len(all_results),      len(test_cases))
    print_metrics("Stanford", avg_metrics(stanford_results), len(stanford_results), len(stanford_test))
    print_metrics("UCSF",     avg_metrics(ucsf_results),     len(ucsf_results),     len(ucsf_test))
    print_metrics("UCLM",     avg_metrics(uclm_results),     len(uclm_results),     len(uclm_test))
    print_metrics("PKG",      avg_metrics(pkg_results),      len(pkg_results),      len(pkg_test))
    print("=" * 72 + "\n")

    output = {
        "overall":  overall,
        "stanford": avg_metrics(stanford_results),
        "ucsf":     avg_metrics(ucsf_results),
        "uclm":     avg_metrics(uclm_results),
        "pkg":      avg_metrics(pkg_results),
        "per_case": {
            "stanford": stanford_results,
            "ucsf":     ucsf_results,
            "uclm":     uclm_results,
            "pkg":      pkg_results,
        },
    }

    out_path = os.path.join(args.output_dir, "evaluation_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
