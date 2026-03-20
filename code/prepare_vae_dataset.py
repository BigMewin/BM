#!/usr/bin/env python3
"""
Prepare Dataset999 split for VAE-MMD pipeline.

Verifies that nnUNet has already preprocessed Dataset999_FourDatasets,
then creates a reproducible 6:2:2 train/val/test split across all four
domains and saves it for downstream use.

Usage:
    python prepare_vae_dataset.py \
        --preprocessed_dir /path/to/nnUNet_preprocessed/Dataset999_FourDatasets/nnUNetPlans_3d_fullres \
        --output_dir /path/to/output \
        [--train_ratio 0.6] \
        [--val_ratio 0.2] \
        [--random_state 42]
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Dataset999 preprocessing and create 6:2:2 split."
    )
    parser.add_argument(
        "--preprocessed_dir", type=str, required=True,
        help="Path to nnUNetPlans_3d_fullres directory containing .b2nd files.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write the split .npz file.",
    )
    parser.add_argument("--train_ratio",  type=float, default=0.6)
    parser.add_argument("--val_ratio",    type=float, default=0.2)
    parser.add_argument("--random_state", type=int,   default=42)
    return parser.parse_args()


def split_domain(
    cases: list,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple:
    """
    Split a list of cases into train / val / test at given ratios.

    Args:
        cases: List of case name stems.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        random_state: Random seed.

    Returns:
        train, val, test lists.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    train, temp = train_test_split(
        cases,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
    )
    val, test = train_test_split(
        temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
    )
    return train, val, test


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.preprocessed_dir):
        logger.error("Preprocessed directory not found: %s", args.preprocessed_dir)
        logger.error("Run nnUNetv2_plan_and_preprocess -d 999 -c 3d_fullres first.")
        sys.exit(1)

    b2nd_files = glob.glob(os.path.join(args.preprocessed_dir, "*.b2nd"))
    all_cases = sorted([
        os.path.basename(f).replace(".b2nd", "")
        for f in b2nd_files
        if "_seg" not in os.path.basename(f)
    ])

    if not all_cases:
        logger.error("No .b2nd files found in %s", args.preprocessed_dir)
        sys.exit(1)

    logger.info("Found %d preprocessed cases", len(all_cases))

    stanford = sorted([c for c in all_cases if c.startswith("stanford_")])
    ucsf     = sorted([c for c in all_cases if c.startswith("ucsf_")])
    uclm     = sorted([c for c in all_cases if c.startswith("uclm_")])
    pkg      = sorted([c for c in all_cases if c.startswith("pkg_")])

    logger.info(
        "Domain counts: Stanford=%d  UCSF=%d  UCLM=%d  PKG=%d",
        len(stanford), len(ucsf), len(uclm), len(pkg),
    )

    splits = {}
    all_train, all_val, all_test = [], [], []

    for name, domain_cases in [
        ("stanford", stanford),
        ("ucsf",     ucsf),
        ("uclm",     uclm),
        ("pkg",      pkg),
    ]:
        tr, va, te = split_domain(
            domain_cases, args.train_ratio, args.val_ratio, args.random_state
        )
        splits[f"{name}_train"] = tr
        splits[f"{name}_val"]   = va
        splits[f"{name}_test"]  = te
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)

        logger.info(
            "%s: %d train / %d val / %d test",
            name.upper(), len(tr), len(va), len(te),
        )

    splits["train_cases"] = all_train
    splits["val_cases"]   = all_val
    splits["test_cases"]  = all_test

    logger.info(
        "Total split: %d train / %d val / %d test",
        len(all_train), len(all_val), len(all_test),
    )

    out_path = os.path.join(args.output_dir, "four_datasets_split.npz")
    np.savez(out_path, **{k: np.array(v) for k, v in splits.items()})
    logger.info("Split saved to %s", out_path)


if __name__ == "__main__":
    main()
