#!/usr/bin/env python3
"""
Reconstruct train and val cases with a trained VAE-MMD model.

Reads .b2nd files from Dataset999 (nnUNet preprocessed), passes each volume
through the VAE encoder-decoder, and writes domain-normalized .b2nd files to
a new Dataset998 directory. Segmentation (.b2nd _seg) and metadata (.pkl)
files are copied verbatim. Test cases are intentionally excluded to preserve
evaluation integrity.

Usage:
    python reconstruct_with_vae.py \
        --checkpoint /path/to/vae_mmd_best.pth \
        --preprocessed_dir /path/to/Dataset999_FourDatasets/nnUNetPlans_3d_fullres \
        --preprocessed_base /path/to/Dataset999_FourDatasets \
        --split_file /path/to/four_datasets_split.npz \
        --output_base /path/to/nnUNet_preprocessed/Dataset998_FourDatasets_VAE \
        [--latent_dim 512]
"""

import argparse
import json
import logging
import os
import shutil
import sys

import blosc2
import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import HighQualityVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BLOSC2_CPARAMS = {"codec": blosc2.Codec.ZSTD, "clevel": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct train/val cases with VAE-MMD."
    )
    parser.add_argument("--checkpoint",        type=str, required=True)
    parser.add_argument("--preprocessed_dir",  type=str, required=True,
                        help="nnUNetPlans_3d_fullres directory of Dataset999.")
    parser.add_argument("--preprocessed_base", type=str, required=True,
                        help="Base directory of Dataset999 (contains nnUNetPlans.json etc).")
    parser.add_argument("--split_file",        type=str, required=True,
                        help="Path to four_datasets_split.npz from prepare_vae_dataset.py.")
    parser.add_argument("--output_base",       type=str, required=True,
                        help="Output base directory for Dataset998.")
    parser.add_argument("--latent_dim",        type=int, default=512)
    return parser.parse_args()


def reconstruct_volume(
    data: np.ndarray,
    vae: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    Pass a single volume through the VAE encoder-decoder.

    The volume is resized to 128^3, normalized to [-1, 1], reconstructed,
    then resized back to the original shape and rescaled to the original
    intensity range.

    Args:
        data: Input volume, shape (D, H, W) or (C, D, H, W).
        vae: Trained HighQualityVAE in eval mode.
        device: Torch device.

    Returns:
        Reconstructed volume with same shape and intensity range as input,
        wrapped in a leading channel dim: (1, D, H, W).
    """
    if data.ndim == 4:
        data = data[0]

    D, H, W = data.shape
    lo, hi = data.min(), data.max()

    data_norm = 2.0 * (data - lo) / (hi - lo + 1e-8) - 1.0
    data_128 = zoom(data_norm, [128.0 / D, 128.0 / H, 128.0 / W], order=1)

    tensor = torch.FloatTensor(data_128[np.newaxis, np.newaxis, ...]).to(device)

    with torch.no_grad():
        recon, _, _, _ = vae(tensor)
        recon_np = recon.cpu().numpy()[0, 0]

    recon_orig = zoom(recon_np, [D / 128.0, H / 128.0, W / 128.0], order=1)
    recon_orig = (recon_orig + 1.0) / 2.0 * (hi - lo) + lo

    return recon_orig[np.newaxis, ...].astype(np.float32)


def copy_config_files(src_base: str, dst_base: str, new_dataset_name: str):
    """
    Copy and patch nnUNet configuration JSON files from Dataset999 to Dataset998.
    """
    for filename in ["nnUNetPlans.json", "dataset.json", "dataset_fingerprint.json"]:
        src = os.path.join(src_base, filename)
        if not os.path.exists(src):
            logger.warning("Config file not found, skipping: %s", src)
            continue
        with open(src) as f:
            config = json.load(f)
        for key in ("dataset_name", "name"):
            if key in config:
                config[key] = new_dataset_name
        dst = os.path.join(dst_base, filename)
        with open(dst, "w") as f:
            json.dump(config, f, indent=2)
    logger.info("Config files copied to %s", dst_base)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading VAE checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    latent_dim = checkpoint.get("config", {}).get("latent_dim", args.latent_dim)
    output_activation = checkpoint.get("config", {}).get("output_activation", "tanh")

    vae = HighQualityVAE(latent_dim=latent_dim, output_activation=output_activation).to(device)
    vae.load_state_dict(checkpoint["vae_state_dict"])
    vae.eval()
    logger.info("VAE loaded (latent_dim=%d)", latent_dim)

    split = np.load(args.split_file, allow_pickle=True)
    train_cases = list(split["train_cases"])
    val_cases   = list(split["val_cases"])
    cases_to_reconstruct = train_cases + val_cases
    logger.info(
        "Reconstructing %d cases (train=%d, val=%d) — test excluded",
        len(cases_to_reconstruct), len(train_cases), len(val_cases),
    )

    output_dir = os.path.join(args.output_base, "nnUNetPlans_3d_fullres")
    if os.path.exists(args.output_base):
        logger.info("Removing existing output directory: %s", args.output_base)
        shutil.rmtree(args.output_base)
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = os.path.basename(args.output_base)
    copy_config_files(args.preprocessed_base, args.output_base, dataset_name)

    failed = []
    for case_name in tqdm(cases_to_reconstruct, desc="VAE reconstruction"):
        data_path = os.path.join(args.preprocessed_dir, f"{case_name}.b2nd")
        seg_path  = os.path.join(args.preprocessed_dir, f"{case_name}_seg.b2nd")
        pkl_path  = os.path.join(args.preprocessed_dir, f"{case_name}.pkl")

        if not os.path.exists(data_path):
            logger.warning("Missing data file, skipping: %s", data_path)
            failed.append(case_name)
            continue

        try:
            data = np.array(blosc2.open(data_path)[:])
            recon = reconstruct_volume(data, vae, device)

            blosc2.asarray(
                recon,
                urlpath=os.path.join(output_dir, f"{case_name}.b2nd"),
                mode="w",
                cparams=BLOSC2_CPARAMS,
            )

            if os.path.exists(seg_path):
                seg = np.array(blosc2.open(seg_path)[:])
                blosc2.asarray(
                    seg,
                    urlpath=os.path.join(output_dir, f"{case_name}_seg.b2nd"),
                    mode="w",
                    cparams=BLOSC2_CPARAMS,
                )

            if os.path.exists(pkl_path):
                shutil.copy(pkl_path, os.path.join(output_dir, f"{case_name}.pkl"))

        except Exception as exc:
            logger.error("Failed to reconstruct %s: %s", case_name, exc)
            failed.append(case_name)

    logger.info(
        "Reconstruction complete: %d successful, %d failed",
        len(cases_to_reconstruct) - len(failed), len(failed),
    )
    if failed:
        logger.warning("Failed cases: %s", failed)


if __name__ == "__main__":
    main()
