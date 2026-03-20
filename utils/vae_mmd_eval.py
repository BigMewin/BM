#!/usr/bin/env python3
"""
Evaluation utilities for VAE-MMD domain adaptation.

Includes:
    - extract_raw_features: 7D intensity statistics from a raw volume
    - extract_features_from_dataset: batch feature extraction (raw + VAE latent)
    - run_tsne: t-SNE dimensionality reduction for raw and VAE features
    - plot_tsne: side-by-side Before/After domain alignment scatter plot
    - evaluate_domain_classifier: logistic regression domain separability test
    - run_full_evaluation: end-to-end eval pipeline
"""

import logging
import os
from typing import Optional

import blosc2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tqdm import tqdm

logger = logging.getLogger(__name__)

DOMAIN_NAMES = ["Stanford", "UCSF", "UCLM", "PKG"]
DOMAIN_COLORS = ["#E63946", "#2A9D8F", "#457B9D", "#E9C46A"]
DOMAIN_MARKERS = ["o", "s", "^", "D"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_raw_features(data: np.ndarray) -> np.ndarray:
    """
    Extract 7D intensity statistics from a raw volume.

    Args:
        data: Array of shape (D, H, W) or (C, D, H, W).

    Returns:
        1D array of shape (7,): [mean, std, min, max, median, p25, p75].
    """
    if data.ndim == 4:
        data = data[0]
    flat = data.flatten()
    return np.array([
        np.mean(flat),
        np.std(flat),
        np.min(flat),
        np.max(flat),
        np.median(flat),
        np.percentile(flat, 25),
        np.percentile(flat, 75),
    ], dtype=np.float32)


def extract_features_from_dataset(
    preprocessed_dir: str,
    case_names: list,
    vae: torch.nn.Module,
    device: torch.device,
) -> tuple:
    """
    Extract raw (7D) and VAE latent (latent_dim) features for all cases.

    Args:
        preprocessed_dir: Directory containing .b2nd files.
        case_names: List of case name stems.
        vae: Trained HighQualityVAE in eval mode.
        device: Torch device.

    Returns:
        raw_features (np.ndarray): Shape [N, 7].
        vae_features (np.ndarray): Shape [N, latent_dim].
        labels (np.ndarray): Integer domain labels, shape [N].
        valid_case_names (list[str]): Case names that were successfully processed.
    """
    from .vae_mmd_data import _infer_domain, DOMAIN_MAP

    raw_features = []
    vae_features = []
    labels = []
    valid_cases = []
    failed = []

    with torch.no_grad():
        for case_name in tqdm(case_names, desc="Extracting features"):
            filepath = os.path.join(preprocessed_dir, f"{case_name}.b2nd")
            if not os.path.exists(filepath):
                failed.append((case_name, "file not found"))
                continue

            domain = _infer_domain(case_name)
            if domain is None:
                failed.append((case_name, "unknown domain"))
                continue

            try:
                data = np.array(blosc2.open(filepath)[:])

                raw_feat = extract_raw_features(data)

                if data.ndim == 4:
                    data = data[0]
                D, H, W = data.shape
                data_128 = zoom(data, [128.0 / D, 128.0 / H, 128.0 / W], order=1)
                lo, hi = data_128.min(), data_128.max()
                data_norm = 2.0 * (data_128 - lo) / (hi - lo + 1e-8) - 1.0
                tensor = torch.FloatTensor(data_norm[np.newaxis, np.newaxis, ...]).to(device)

                mu, _, _ = vae.encode(tensor)
                vae_feat = mu.cpu().numpy().flatten()

                raw_features.append(raw_feat)
                vae_features.append(vae_feat)
                labels.append(domain)
                valid_cases.append(case_name)

            except Exception as exc:
                failed.append((case_name, str(exc)))
                continue

    if failed:
        logger.warning("%d cases failed during feature extraction:", len(failed))
        for name, reason in failed[:10]:
            logger.warning("  %s: %s", name, reason)

    logger.info(
        "Feature extraction complete: %d successful, %d failed",
        len(valid_cases), len(failed),
    )

    return (
        np.array(raw_features, dtype=np.float32),
        np.array(vae_features, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        valid_cases,
    )


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def run_tsne(
    raw_features: np.ndarray,
    vae_features: np.ndarray,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> tuple:
    """
    Run t-SNE on raw and VAE features.

    Args:
        raw_features: Shape [N, 7].
        vae_features: Shape [N, latent_dim].
        random_state: Random seed for reproducibility.
        perplexity: t-SNE perplexity parameter.

    Returns:
        raw_tsne (np.ndarray): Shape [N, 2].
        vae_tsne (np.ndarray): Shape [N, 2].
    """
    scaler_raw = StandardScaler()
    X_raw = scaler_raw.fit_transform(raw_features)
    logger.info("Running t-SNE on raw features (%s)...", raw_features.shape)
    raw_tsne = TSNE(
        n_components=2, random_state=random_state, perplexity=perplexity
    ).fit_transform(X_raw)

    scaler_vae = StandardScaler()
    X_vae = scaler_vae.fit_transform(vae_features)
    logger.info("Running t-SNE on VAE features (%s)...", vae_features.shape)
    vae_tsne = TSNE(
        n_components=2, random_state=random_state, perplexity=perplexity
    ).fit_transform(X_vae)

    return raw_tsne, vae_tsne


def plot_tsne(
    raw_tsne: np.ndarray,
    vae_tsne: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    dpi: int = 200,
):
    """
    Save a side-by-side Before/After t-SNE scatter plot.

    Args:
        raw_tsne: t-SNE coordinates for raw features, shape [N, 2].
        vae_tsne: t-SNE coordinates for VAE features, shape [N, 2].
        labels: Domain labels, shape [N].
        output_path: File path to save the PNG.
        dpi: Output resolution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, coords, title in zip(
        axes,
        [raw_tsne, vae_tsne],
        ["Before VAE-MMD (Raw Features)", "After VAE-MMD (Latent Features)"],
    ):
        for i, (name, color, marker) in enumerate(
            zip(DOMAIN_NAMES, DOMAIN_COLORS, DOMAIN_MARKERS)
        ):
            mask = labels == i
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, marker=marker, s=100, alpha=0.7,
                label=name, edgecolors="black", linewidth=0.5,
            )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("t-SNE plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Domain classifier
# ---------------------------------------------------------------------------

def evaluate_domain_classifier(
    raw_features: np.ndarray,
    vae_features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.3,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train a logistic regression classifier on raw and VAE features
    to quantify domain separability before and after VAE-MMD alignment.

    Lower accuracy after VAE-MMD indicates better domain mixing.

    Args:
        raw_features: Shape [N, 7].
        vae_features: Shape [N, latent_dim].
        labels: Integer domain labels, shape [N].
        test_size: Fraction of data held out for testing.
        cv_folds: Number of cross-validation folds.
        random_state: Random seed.

    Returns:
        Dictionary with keys:
            raw_cv_mean, raw_cv_std, raw_test_acc,
            vae_cv_mean, vae_cv_std, vae_test_acc,
            domain_confusion_pct,
            raw_report, vae_report,
            raw_cm, vae_cm  (confusion matrices as np.ndarray).
    """
    scaler_raw = StandardScaler()
    X_raw = scaler_raw.fit_transform(raw_features)

    scaler_vae = StandardScaler()
    X_vae = scaler_vae.fit_transform(vae_features)

    X_raw_tr, X_raw_te, y_tr, y_te = train_test_split(
        X_raw, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    X_vae_tr, X_vae_te, _, _ = train_test_split(
        X_vae, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    clf_raw = LogisticRegression(max_iter=2000, random_state=random_state)
    clf_raw.fit(X_raw_tr, y_tr)
    raw_cv = cross_val_score(clf_raw, X_raw, labels, cv=cv_folds, scoring="accuracy")
    raw_test_acc = clf_raw.score(X_raw_te, y_te)
    y_pred_raw = clf_raw.predict(X_raw_te)

    clf_vae = LogisticRegression(max_iter=2000, random_state=random_state)
    clf_vae.fit(X_vae_tr, y_tr)
    vae_cv = cross_val_score(clf_vae, X_vae, labels, cv=cv_folds, scoring="accuracy")
    vae_test_acc = clf_vae.score(X_vae_te, y_te)
    y_pred_vae = clf_vae.predict(X_vae_te)

    random_baseline = 1.0 / len(np.unique(labels))
    acc_drop = raw_cv.mean() - vae_cv.mean()
    domain_confusion_pct = (
        acc_drop / (raw_cv.mean() - random_baseline) * 100.0
        if (raw_cv.mean() - random_baseline) > 0
        else 0.0
    )

    logger.info("--- Domain Classifier Results ---")
    logger.info("Random baseline:   %.4f", random_baseline)
    logger.info("Before VAE-MMD CV: %.4f +/- %.4f", raw_cv.mean(), raw_cv.std())
    logger.info("After  VAE-MMD CV: %.4f +/- %.4f", vae_cv.mean(), vae_cv.std())
    logger.info("Accuracy drop:     %.4f", acc_drop)
    logger.info("Domain confusion:  %.1f%%", domain_confusion_pct)

    return {
        "raw_cv_mean":           raw_cv.mean(),
        "raw_cv_std":            raw_cv.std(),
        "raw_test_acc":          raw_test_acc,
        "vae_cv_mean":           vae_cv.mean(),
        "vae_cv_std":            vae_cv.std(),
        "vae_test_acc":          vae_test_acc,
        "domain_confusion_pct":  domain_confusion_pct,
        "random_baseline":       random_baseline,
        "raw_report":            classification_report(
                                     y_te, y_pred_raw,
                                     target_names=DOMAIN_NAMES, zero_division=0
                                 ),
        "vae_report":            classification_report(
                                     y_te, y_pred_vae,
                                     target_names=DOMAIN_NAMES, zero_division=0
                                 ),
        "raw_cm":                confusion_matrix(y_te, y_pred_raw),
        "vae_cm":                confusion_matrix(y_te, y_pred_vae),
    }


def plot_confusion_matrices(
    results: dict,
    output_path: str,
    dpi: int = 300,
):
    """
    Save side-by-side confusion matrix heatmaps for Before/After VAE-MMD.

    Args:
        results: Output dict from evaluate_domain_classifier().
        output_path: File path to save the PNG.
        dpi: Output resolution.
    """
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, cm, title, acc, cmap in zip(
        axes,
        [results["raw_cm"], results["vae_cm"]],
        ["Before VAE-MMD", "After VAE-MMD"],
        [results["raw_test_acc"], results["vae_test_acc"]],
        ["Blues", "Greens"],
    ):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap,
            xticklabels=DOMAIN_NAMES, yticklabels=DOMAIN_NAMES,
            ax=ax, cbar_kws={"label": "Count"},
        )
        ax.set_title(f"{title}\nAccuracy: {acc:.3f}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrices saved to %s", output_path)


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    preprocessed_dir: str,
    case_names: list,
    vae: torch.nn.Module,
    device: torch.device,
    output_dir: str,
    tsne_perplexity: float = 30.0,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Run the complete domain adaptation evaluation pipeline:
        1. Extract raw and VAE latent features for all cases.
        2. Run t-SNE and save scatter plot.
        3. Train domain classifier and save confusion matrices.
        4. Print and return a summary.

    Args:
        preprocessed_dir: Directory containing .b2nd files.
        case_names: List of case name stems.
        vae: Trained HighQualityVAE in eval mode.
        device: Torch device.
        output_dir: Directory to write output figures.
        tsne_perplexity: t-SNE perplexity.
        cv_folds: Cross-validation folds for domain classifier.
        random_state: Random seed.

    Returns:
        Dictionary with classifier results and output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_feat, vae_feat, labels, valid_cases = extract_features_from_dataset(
        preprocessed_dir, case_names, vae, device
    )

    raw_tsne, vae_tsne = run_tsne(
        raw_feat, vae_feat,
        random_state=random_state,
        perplexity=tsne_perplexity,
    )
    tsne_path = os.path.join(output_dir, "tsne_domain_alignment.png")
    plot_tsne(raw_tsne, vae_tsne, labels, tsne_path)

    clf_results = evaluate_domain_classifier(
        raw_feat, vae_feat, labels,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrices(clf_results, cm_path)

    print("\n" + "=" * 72)
    print("Domain Adaptation Evaluation Summary")
    print("=" * 72)
    print(f"Samples evaluated:   {len(valid_cases)}")
    print(f"Random baseline:     {clf_results['random_baseline']:.4f}")
    print(f"Before VAE-MMD  CV:  {clf_results['raw_cv_mean']:.4f} +/- {clf_results['raw_cv_std']:.4f}")
    print(f"After  VAE-MMD  CV:  {clf_results['vae_cv_mean']:.4f} +/- {clf_results['vae_cv_std']:.4f}")
    print(f"Accuracy drop:       {clf_results['raw_cv_mean'] - clf_results['vae_cv_mean']:.4f}")
    print(f"Domain confusion:    {clf_results['domain_confusion_pct']:.1f}%")

    threshold_excellent = 0.35
    threshold_good = 0.50
    if clf_results["vae_cv_mean"] < threshold_excellent:
        verdict = "Excellent: VAE successfully mixed domain features; latent space is well-aligned."
    elif clf_results["vae_cv_mean"] < threshold_good:
        verdict = "Good: VAE substantially reduced domain separability."
    else:
        verdict = "Limited: domain adaptation effect is marginal."
    print(f"Verdict:             {verdict}")
    print("=" * 72 + "\n")

    print("Before VAE-MMD classification report:")
    print(clf_results["raw_report"])
    print("After VAE-MMD classification report:")
    print(clf_results["vae_report"])

    clf_results["tsne_plot"] = tsne_path
    clf_results["confusion_matrix_plot"] = cm_path

    return clf_results
