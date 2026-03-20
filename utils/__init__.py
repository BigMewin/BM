from .vae_mmd_model import ResidualBlock3D, HighQualityVAE, Discriminator3D
from .vae_mmd_losses import compute_mmd, ssim_3d, compute_vae_loss, compute_disc_loss
from .vae_mmd_data import FourDatasetPreprocessed, load_blosc2_array, resize_to_128, DOMAIN_MAP
from .vae_mmd_eval import (
    extract_raw_features,
    extract_features_from_dataset,
    run_tsne,
    plot_tsne,
    evaluate_domain_classifier,
    plot_confusion_matrices,
    run_full_evaluation,
)
from .vae_mmd_metrics import (
    compute_surface_distances,
    compute_hd95,
    compute_surface_dice,
    compute_metrics,
    avg_metrics,
    print_metrics,
)

__all__ = [
    # VAE components
    "ResidualBlock3D",
    "HighQualityVAE",
    "Discriminator3D",
    # Losses
    "compute_mmd",
    "ssim_3d",
    "compute_vae_loss",
    "compute_disc_loss",
    # Data
    "FourDatasetPreprocessed",
    "load_blosc2_array",
    "resize_to_128",
    "DOMAIN_MAP",
    # Evaluation - domain adaptation
    "extract_raw_features",
    "extract_features_from_dataset",
    "run_tsne",
    "plot_tsne",
    "evaluate_domain_classifier",
    "plot_confusion_matrices",
    "run_full_evaluation",
    # Evaluation - segmentation metrics
    "compute_surface_distances",
    "compute_hd95",
    "compute_surface_dice",
    "compute_metrics",
    "avg_metrics",
    "print_metrics",
]
