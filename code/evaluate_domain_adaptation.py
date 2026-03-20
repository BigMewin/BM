#!/usr/bin/env python3
"""
Evaluate VAE-MMD domain adaptation quality.

Loads a trained VAE checkpoint, extracts raw and latent features for all
preprocessed cases, runs t-SNE visualization, and trains a logistic regression
domain classifier to quantify domain alignment.

Usage:
    python evaluate_domain_adaptation.py \
        --checkpoint /path/to/vae_mmd_best.pth \
        --preprocessed_dir /path/to/nnUNetPlans_3d_fullres \
        --output_dir /path/to/eval_output \
        [--latent_dim 512] \
        [--tsne_perplexity 30] \
        [--cv_folds 5]
"""

import argparse
import glob
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import HighQualityVAE
from utils.vae_mmd_eval import run_full_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VAE-MMD domain adaptation quality."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to VAE checkpoint (.pth) saved by train_vae_mmd_adapter.py.",
    )
    parser.add_argument(
        "--preprocessed_dir", type=str, required=True,
        help="Path to nnUNet 3d_fullres preprocessed directory containing .b2nd files.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write t-SNE plot and confusion matrix figures.",
    )
    parser.add_argument("--latent_dim",       type=int,   default=512)
    parser.add_argument("--tsne_perplexity",  type=float, default=30.0)
    parser.add_argument("--cv_folds",         type=int,   default=5)
    parser.add_argument("--random_state",     type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading VAE checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    latent_dim = (
        checkpoint.get("config", {}).get("latent_dim", args.latent_dim)
    )
    output_activation = (
        checkpoint.get("config", {}).get("output_activation", "tanh")
    )

    vae = HighQualityVAE(latent_dim=latent_dim, output_activation=output_activation).to(device)
    vae.load_state_dict(checkpoint["vae_state_dict"])
    vae.eval()
    logger.info("VAE loaded (latent_dim=%d, output_activation=%s)", latent_dim, output_activation)

    b2nd_files = glob.glob(os.path.join(args.preprocessed_dir, "*.b2nd"))
    case_names = [
        os.path.basename(f).replace(".b2nd", "")
        for f in b2nd_files
        if "_seg" not in os.path.basename(f)
    ]

    if not case_names:
        logger.error("No .b2nd files found in %s", args.preprocessed_dir)
        sys.exit(1)

    logger.info("Found %d cases for evaluation", len(case_names))

    run_full_evaluation(
        preprocessed_dir=args.preprocessed_dir,
        case_names=case_names,
        vae=vae,
        device=device,
        output_dir=args.output_dir,
        tsne_perplexity=args.tsne_perplexity,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
