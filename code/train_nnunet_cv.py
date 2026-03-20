#!/usr/bin/env python3
"""
Generate 5-fold cross-validation splits and train nnUNet on VAE-reconstructed data.

Reads the train/val cases from the split file, creates a splits_final.json
with 5 folds (test set is excluded from all folds), then launches nnUNet
training for each fold sequentially.

Usage:
    python train_nnunet_cv.py \
        --preprocessed_base /path/to/nnUNet_preprocessed/Dataset998_FourDatasets_VAE \
        --split_file /path/to/four_datasets_split.npz \
        --dataset_id 998 \
        [--folds 0 1 2 3 4] \
        [--nnunet_results /path/to/nnUNet_results]
"""

import argparse
import json
import logging
import os
import subprocess
import sys

import numpy as np
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 5-fold CV splits and train nnUNet on VAE data."
    )
    parser.add_argument("--preprocessed_base", type=str, required=True,
                        help="Base directory of Dataset998 (contains nnUNetPlans.json).")
    parser.add_argument("--split_file",        type=str, required=True,
                        help="Path to four_datasets_split.npz.")
    parser.add_argument("--dataset_id",        type=int, default=998)
    parser.add_argument("--folds",             type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--nnunet_results",    type=str,
                        default=os.environ.get("nnUNet_results", "/tmp/nnUNet_results"))
    parser.add_argument("--n_splits",          type=int, default=5)
    return parser.parse_args()


def create_cv_splits(
    train_cases: list,
    val_cases: list,
    n_splits: int,
    random_state: int = 42,
) -> list:
    """
    Create n_splits cross-validation folds from train+val cases.

    Each fold uses (n_splits - 1) / n_splits of train+val as training
    and 1 / n_splits as validation. Test cases are never included.

    Args:
        train_cases: Training case names.
        val_cases: Validation case names.
        n_splits: Number of CV folds.
        random_state: Random seed for KFold shuffle.

    Returns:
        List of dicts with keys 'train' and 'val', one per fold.
    """
    all_trainval = train_cases + val_cases
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits = []
    for train_idx, val_idx in kf.split(all_trainval):
        fold_train = [all_trainval[i] for i in train_idx]
        fold_val   = [all_trainval[i] for i in val_idx]
        splits.append({"train": fold_train, "val": fold_val})

    return splits


def run_training(dataset_id: int, fold: int, nnunet_results: str):
    """
    Launch nnUNetv2_train for a single fold as a subprocess.

    Args:
        dataset_id: nnUNet dataset ID.
        fold: Fold index (0-4).
        nnunet_results: Path to nnUNet_results directory.
    """
    env = os.environ.copy()
    env["nnUNet_results"] = nnunet_results

    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        "3d_fullres",
        str(fold),
        "--npz",
    ]

    logger.info("Training fold %d: %s", fold, " ".join(cmd))
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        logger.error("nnUNet training failed for fold %d (exit code %d)", fold, result.returncode)
        sys.exit(result.returncode)

    logger.info("Fold %d training complete", fold)


def main():
    args = parse_args()

    split = np.load(args.split_file, allow_pickle=True)
    train_cases = list(split["train_cases"])
    val_cases   = list(split["val_cases"])
    test_cases  = list(split["test_cases"])

    logger.info(
        "Split loaded: train=%d  val=%d  test=%d (excluded from all folds)",
        len(train_cases), len(val_cases), len(test_cases),
    )

    cv_splits = create_cv_splits(train_cases, val_cases, n_splits=args.n_splits)

    splits_path = os.path.join(args.preprocessed_base, "splits_final.json")
    with open(splits_path, "w") as f:
        json.dump(cv_splits, f, indent=4)

    logger.info(
        "splits_final.json written to %s (%d folds)", splits_path, len(cv_splits)
    )
    for i, fold in enumerate(cv_splits):
        logger.info(
            "  Fold %d: %d train / %d val",
            i, len(fold["train"]), len(fold["val"]),
        )

    os.makedirs(args.nnunet_results, exist_ok=True)

    for fold in args.folds:
        if fold >= args.n_splits:
            logger.error(
                "Fold %d requested but only %d folds defined", fold, args.n_splits
            )
            sys.exit(1)
        run_training(args.dataset_id, fold, args.nnunet_results)

    logger.info("All folds trained: %s", args.folds)


if __name__ == "__main__":
    main()
