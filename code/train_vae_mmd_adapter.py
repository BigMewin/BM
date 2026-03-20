#!/usr/bin/env python3
"""
VAE-MMD training script for brain metastasis domain adaptation.

Trains a HighQualityVAE with MMD regularization across four datasets:
Stanford, UCSF, UCLM, PKG.

Usage:
    python train_vae_mmd_adapter.py --preprocessed_dir /path/to/nnUNet_preprocessed/Dataset999_FourDatasets/nnUNetPlans_3d_fullres
                            --output_dir /path/to/output
                            [--latent_dim 512]
                            [--batch_size 4]
                            [--num_epochs 100]
                            [--lr 1e-4]
                            [--save_interval 10]

Environment:
    nnUNet_raw, nnUNet_preprocessed, nnUNet_results must be set before
    running nnUNetv2_plan_and_preprocess to generate the .b2nd files
    consumed by this script.
"""

import argparse
import glob
import logging
import os
import shutil
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    HighQualityVAE,
    Discriminator3D,
    FourDatasetPreprocessed,
    compute_vae_loss,
    compute_disc_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VAE-MMD for brain metastasis domain adaptation."
    )
    parser.add_argument(
        "--preprocessed_dir", type=str, required=True,
        help="Path to nnUNet 3d_fullres preprocessed directory containing .b2nd files.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write checkpoints, final model, and training curves.",
    )
    parser.add_argument("--latent_dim",    type=int,   default=512)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--num_epochs",    type=int,   default=100)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int,   default=10,
                        help="Save a checkpoint every N epochs.")

    # Loss weights
    parser.add_argument("--lambda_mse",  type=float, default=300.0)
    parser.add_argument("--lambda_l1",   type=float, default=150.0)
    parser.add_argument("--lambda_ssim", type=float, default=50.0)
    parser.add_argument("--lambda_mmd",  type=float, default=10.0)
    parser.add_argument("--lambda_kl",   type=float, default=0.1)
    parser.add_argument("--lambda_adv",  type=float, default=5.0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, epoch: int, vae, disc, vae_opt, disc_opt, loss: float, config: dict):
    torch.save(
        {
            "epoch":                    epoch,
            "vae_state_dict":           vae.state_dict(),
            "discriminator_state_dict": disc.state_dict(),
            "vae_optimizer":            vae_opt.state_dict(),
            "disc_optimizer":           disc_opt.state_dict(),
            "loss":                     loss,
            "config":                   config,
        },
        path,
    )


def save_history_checkpoint(path: str, epoch: int, vae, disc, history: dict):
    torch.save(
        {
            "epoch":                    epoch,
            "vae_state_dict":           vae.state_dict(),
            "discriminator_state_dict": disc.state_dict(),
            "history":                  history,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, output_path: str):
    keys = ["vae_loss", "recon_loss", "mmd_loss", "kl_loss", "adv_loss", "disc_loss"]
    titles = [
        "Total VAE Loss",
        "Reconstruction Loss",
        "MMD Loss (Domain Adaptation)",
        "KL Divergence",
        "Adversarial Loss (VAE)",
        "Discriminator Loss",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, key, title in zip(axes.flat, keys, titles):
        ax.plot(history[key])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Discover preprocessed cases
    b2nd_files = glob.glob(os.path.join(args.preprocessed_dir, "*.b2nd"))
    all_cases = [
        os.path.basename(f).replace(".b2nd", "")
        for f in b2nd_files
        if "_seg" not in os.path.basename(f)
    ]

    if not all_cases:
        logger.error("No .b2nd files found in %s", args.preprocessed_dir)
        sys.exit(1)

    logger.info("Found %d preprocessed cases", len(all_cases))

    # Dataset and dataloader
    dataset = FourDatasetPreprocessed(
        preprocessed_dir=args.preprocessed_dir,
        case_names=all_cases,
        augment=True,
        normalization="minmax",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Models
    vae           = HighQualityVAE(latent_dim=args.latent_dim, output_activation="tanh").to(device)
    discriminator = Discriminator3D().to(device)

    vae_params  = sum(p.numel() for p in vae.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    logger.info("VAE parameters:           %s", f"{vae_params:,}")
    logger.info("Discriminator parameters: %s", f"{disc_params:,}")

    # Optimizers
    vae_optimizer  = torch.optim.Adam(vae.parameters(),           lr=args.lr, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Loss config dict (also stored in best checkpoint)
    loss_config = {
        "latent_dim":          args.latent_dim,
        "output_activation":   "tanh",
        "normalization":       "minmax",
        "lambda_recon_mse":    args.lambda_mse,
        "lambda_recon_l1":     args.lambda_l1,
        "lambda_recon_ssim":   args.lambda_ssim,
        "lambda_mmd":          args.lambda_mmd,
        "lambda_kl":           args.lambda_kl,
        "lambda_adv":          args.lambda_adv,
    }

    logger.info("Loss weights: %s", loss_config)

    # Training history
    history = defaultdict(list)

    best_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, "vae_mmd_best.pth")

    for epoch in range(args.num_epochs):
        vae.train()
        discriminator.train()

        epoch_losses = defaultdict(float)
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)

        for batch, labels in pbar:
            batch  = batch.to(device)
            labels = labels.to(device)

            # --- VAE update ---
            vae_optimizer.zero_grad()

            recon, mu, logvar, z = vae(batch)

            vae_loss, components = compute_vae_loss(
                recon=recon,
                target=batch,
                mu=mu,
                logvar=logvar,
                z=z,
                domain_labels=labels,
                discriminator=discriminator,
                lambda_recon_mse=args.lambda_mse,
                lambda_recon_l1=args.lambda_l1,
                lambda_recon_ssim=args.lambda_ssim,
                lambda_mmd=args.lambda_mmd,
                lambda_kl=args.lambda_kl,
                lambda_adv=args.lambda_adv,
            )

            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            vae_optimizer.step()

            # --- Discriminator update ---
            disc_optimizer.zero_grad()
            disc_loss = compute_disc_loss(discriminator, batch, recon)
            disc_loss.backward()
            disc_optimizer.step()

            # Accumulate for epoch average
            epoch_losses["vae"]  += vae_loss.item()
            epoch_losses["disc"] += disc_loss.item()
            for k, v in components.items():
                epoch_losses[k] += v
            n_batches += 1

            pbar.set_postfix({
                "VAE":   f"{vae_loss.item():.3f}",
                "Recon": f"{components['recon_total']:.3f}",
                "MMD":   f"{components['mmd']:.4f}",
            })

            if n_batches % 10 == 0:
                torch.cuda.empty_cache()

        # Normalize to per-epoch averages
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Log and record history
        history["vae_loss"].append(epoch_losses["vae"])
        history["recon_loss"].append(epoch_losses["recon_total"])
        history["mmd_loss"].append(epoch_losses["mmd"])
        history["kl_loss"].append(epoch_losses["kl"])
        history["adv_loss"].append(epoch_losses["adv_vae"])
        history["disc_loss"].append(epoch_losses["disc"])

        logger.info(
            "Epoch %d/%d | VAE %.4f | Recon %.4f (MSE %.4f L1 %.4f SSIM %.4f) "
            "| MMD %.4f | KL %.4f | Disc %.4f",
            epoch + 1, args.num_epochs,
            epoch_losses["vae"],
            epoch_losses["recon_total"],
            epoch_losses["recon_mse"],
            epoch_losses["recon_l1"],
            epoch_losses["recon_ssim"],
            epoch_losses["mmd"],
            epoch_losses["kl"],
            epoch_losses["disc"],
        )

        # Save best model
        if epoch_losses["vae"] < best_loss:
            best_loss = epoch_losses["vae"]
            save_checkpoint(
                best_model_path,
                epoch, vae, discriminator,
                vae_optimizer, disc_optimizer,
                best_loss, loss_config,
            )
            logger.info("Best model updated (loss %.4f) -> %s", best_loss, best_model_path)

        # Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"vae_mmd_epoch_{epoch + 1}.pth")
            save_history_checkpoint(ckpt_path, epoch, vae, discriminator, dict(history))
            logger.info("Checkpoint saved: %s", ckpt_path)

    # Final model
    final_path = os.path.join(args.output_dir, "vae_mmd_final.pth")
    torch.save(
        {
            "vae_state_dict":           vae.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "history":                  dict(history),
            "config":                   loss_config,
        },
        final_path,
    )
    logger.info("Final model saved: %s", final_path)

    # Training curves
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(dict(history), curves_path)

    logger.info(
        "Training complete. Best VAE loss: %.4f | Epochs: %d | Samples: %d",
        best_loss, args.num_epochs, len(dataset),
    )


if __name__ == "__main__":
    main()
