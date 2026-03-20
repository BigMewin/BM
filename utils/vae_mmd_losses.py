#!/usr/bin/env python3
"""
Loss functions for VAE-MMD domain adaptation training.

Includes:
    - compute_mmd: multi-scale RBF kernel MMD between two distributions
    - ssim_3d: 3D structural similarity index
    - compute_vae_loss: composite loss combining recon + KL + MMD + adversarial
"""

from typing import Optional
import torch
import torch.nn.functional as F


def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_scales: list = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy with multi-scale RBF kernel.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Args:
        x: Samples from distribution P, shape [n, d].
        y: Samples from distribution Q, shape [m, d].
        kernel_scales: RBF bandwidth parameters (sigma).

    Returns:
        Scalar MMD value averaged over all kernel scales.
    """
    n, m = x.size(0), y.size(0)
    mmd = torch.zeros(1, device=x.device)

    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand(n, n)
    ry = yy.diag().unsqueeze(0).expand(m, m)

    rx_expand = xx.diag().unsqueeze(1).expand(n, m)
    ry_expand = yy.diag().unsqueeze(0).expand(n, m)

    for sigma in kernel_scales:
        denom = 2.0 * sigma ** 2
        K_xx = torch.exp(-(rx + rx.t() - 2 * xx) / denom)
        K_yy = torch.exp(-(ry + ry.t() - 2 * yy) / denom)
        K_xy = torch.exp(-(rx_expand + ry_expand - 2 * xy) / denom)
        mmd = mmd + K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()

    return mmd / len(kernel_scales)


def ssim_3d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    3D Structural Similarity Index (SSIM) using average-pooling approximation.

    Args:
        img1: Tensor of shape [B, 1, D, H, W].
        img2: Tensor of shape [B, 1, D, H, W].
        window_size: Size of the averaging window.
        size_average: If True, return scalar mean; otherwise return per-sample values.

    Returns:
        SSIM value in [0, 1]. Higher is more similar.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    mu1 = F.avg_pool3d(img1, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool3d(img2, window_size, stride=1, padding=pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool3d(img1 * img1, window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = F.avg_pool3d(img2 * img2, window_size, stride=1, padding=pad) - mu2_sq
    sigma12   = F.avg_pool3d(img1 * img2, window_size, stride=1, padding=pad) - mu1_mu2

    ssim_map = (
        (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=[1, 2, 3, 4])


def compute_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    domain_labels: torch.Tensor,
    discriminator: torch.nn.Module,
    lambda_recon_mse: float = 300.0,
    lambda_recon_l1: float = 150.0,
    lambda_recon_ssim: float = 50.0,
    lambda_mmd: float = 10.0,
    lambda_kl: float = 0.1,
    lambda_adv: float = 5.0,
    kernel_scales: tuple = (0.5, 1.0, 2.0),
):
    """
    Composite VAE loss: reconstruction + KL divergence + MMD + adversarial.

    Args:
        recon: Reconstructed volume [B, 1, D, H, W].
        target: Ground-truth volume [B, 1, D, H, W].
        mu: Latent mean [B, latent_dim].
        logvar: Latent log-variance [B, latent_dim].
        z: Latent sample [B, latent_dim] (unused; reserved for extensions).
        domain_labels: Integer domain index per sample [B].
        discriminator: Trained discriminator for adversarial loss.
        lambda_*: Loss weight scalars.
        kernel_scales: RBF kernel bandwidths for MMD.

    Returns:
        total_loss (Tensor): Scalar total VAE loss.
        components (dict): Individual loss terms (detached floats) for logging.
    """
    # Reconstruction losses
    recon_mse  = F.mse_loss(recon, target)
    recon_l1   = F.l1_loss(recon, target)
    recon_ssim = 1.0 - ssim_3d(recon, target)

    recon_loss = (
        lambda_recon_mse  * recon_mse  +
        lambda_recon_l1   * recon_l1   +
        lambda_recon_ssim * recon_ssim
    )

    # KL divergence
    kl_loss = (
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    )

    # MMD across all domain pairs
    mmd_loss = torch.zeros(1, device=mu.device)
    unique_domains = torch.unique(domain_labels)

    if unique_domains.numel() >= 2:
        n_pairs = 0
        for i in range(unique_domains.numel()):
            for j in range(i + 1, unique_domains.numel()):
                mask_i = domain_labels == unique_domains[i]
                mask_j = domain_labels == unique_domains[j]
                if mask_i.sum() > 0 and mask_j.sum() > 0:
                    mmd_loss = mmd_loss + compute_mmd(
                        mu[mask_i], mu[mask_j], kernel_scales=list(kernel_scales)
                    )
                    n_pairs += 1
        if n_pairs > 0:
            mmd_loss = mmd_loss / n_pairs

    # Adversarial loss: VAE tries to fool discriminator
    disc_fake = discriminator(recon)
    adv_loss = F.mse_loss(disc_fake, torch.ones_like(disc_fake))

    total_loss = (
        recon_loss          +
        lambda_kl  * kl_loss  +
        lambda_mmd * mmd_loss +
        lambda_adv * adv_loss
    )

    components = {
        "recon_total": recon_loss.item(),
        "recon_mse":   recon_mse.item(),
        "recon_l1":    recon_l1.item(),
        "recon_ssim":  recon_ssim.item(),
        "kl":          kl_loss.item(),
        "mmd":         mmd_loss.item(),
        "adv_vae":     adv_loss.item(),
    }

    return total_loss, components


def compute_disc_loss(
    discriminator: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """
    LSGAN discriminator loss: (D(real)-1)^2 + D(fake)^2, averaged.

    Args:
        discriminator: The discriminator network.
        real: Real samples [B, 1, D, H, W].
        fake: Reconstructed samples [B, 1, D, H, W] (detached).

    Returns:
        Scalar discriminator loss.
    """
    pred_real = discriminator(real)
    pred_fake = discriminator(fake.detach())

    loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
    loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

    return (loss_real + loss_fake) * 0.5
