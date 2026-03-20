#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for VAE-MMD training.

Includes:
    - DOMAIN_MAP: dataset name -> integer domain label
    - load_blosc2_array: load nnU-Net preprocessed .b2nd files
    - resize_to_128: resample a 3D volume to 128^3
    - FourDatasetPreprocessed: PyTorch Dataset for Stanford/UCSF/UCLM/PKG
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


DOMAIN_MAP = {
    "stanford": 0,
    "ucsf":     1,
    "uclm":     2,
    "pkg":      3,
}


def load_blosc2_array(filepath: str) -> np.ndarray:
    """
    Load a nnU-Net preprocessed blosc2-compressed array from disk.

    Args:
        filepath: Path to a .b2nd file.

    Returns:
        NumPy array with the decompressed data.
    """
    import blosc2
    arr = blosc2.open(filepath)
    return np.array(arr[:])


def resize_to_128(data: np.ndarray) -> np.ndarray:
    """
    Resample a 3D (or 4D with leading channel) volume to 128^3.

    Uses trilinear interpolation (order=1).

    Args:
        data: Array of shape (D, H, W) or (C, D, H, W).

    Returns:
        Array of shape (128, 128, 128).
    """
    if data.ndim == 4:
        data = data[0]

    D, H, W = data.shape
    zoom_factors = [128.0 / D, 128.0 / H, 128.0 / W]
    return zoom(data, zoom_factors, order=1)


def _infer_domain(case_name: str) -> Optional[int]:
    """
    Infer integer domain label from a case name string.

    Args:
        case_name: Filename stem, expected to start with 'stanford_', 'ucsf_', etc.

    Returns:
        Domain integer, or None if unrecognized.
    """
    lower = case_name.lower()
    for key, label in DOMAIN_MAP.items():
        if lower.startswith(key):
            return label
    return None


class FourDatasetPreprocessed(Dataset):
    """
    PyTorch Dataset for nnU-Net preprocessed brain metastasis data.

    Loads .b2nd files from a single flat directory, infers domain labels
    from filename prefixes (stanford_, ucsf_, uclm_, pkg_), and applies
    optional normalization and augmentation.

    Args:
        preprocessed_dir (str): Directory containing .b2nd case files.
        case_names (list[str]): List of case name stems (no extension, no _seg suffix).
        augment (bool): Enable random flip/rotation augmentation.
        normalization (str): 'minmax' -> [-1, 1] for tanh output;
                             'zscore' -> zero mean, unit variance;
                             'none'   -> raw data.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        case_names: list,
        augment: bool = True,
        normalization: str = "minmax",
    ):
        self.preprocessed_dir = preprocessed_dir
        self.augment = augment
        self.normalization = normalization
        self.valid_cases = []

        for case in case_names:
            filepath = os.path.join(preprocessed_dir, f"{case}.b2nd")
            if not os.path.exists(filepath):
                logger.debug("Skipping missing file: %s", filepath)
                continue

            domain = _infer_domain(case)
            if domain is None:
                logger.warning("Could not determine domain for case: %s", case)
                continue

            self.valid_cases.append({
                "name":     case,
                "domain":   domain,
                "filepath": filepath,
            })

        domain_counts = {k: 0 for k in DOMAIN_MAP}
        for c in self.valid_cases:
            for key, label in DOMAIN_MAP.items():
                if c["domain"] == label:
                    domain_counts[key] += 1

        logger.info(
            "Dataset initialized: %d cases | %s",
            len(self.valid_cases),
            " | ".join(f"{k.upper()}={v}" for k, v in domain_counts.items()),
        )

    def __len__(self) -> int:
        return len(self.valid_cases)

    def __getitem__(self, idx: int):
        """
        Returns:
            image (Tensor): Shape [1, 128, 128, 128], normalized.
            domain (int): Integer domain label.
        """
        case = self.valid_cases[idx]
        data = load_blosc2_array(case["filepath"])
        data = resize_to_128(data)

        if self.normalization == "minmax":
            lo, hi = data.min(), data.max()
            data = (data - lo) / (hi - lo + 1e-8)
            data = 2.0 * data - 1.0
        elif self.normalization == "zscore":
            data = (data - data.mean()) / (data.std() + 1e-8)

        if self.augment:
            for axis in range(3):
                if random.random() > 0.5:
                    data = np.flip(data, axis=axis).copy()
            if random.random() > 0.7:
                k = random.randint(1, 3)
                data = np.rot90(data, k, axes=(0, 1)).copy()

        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        return torch.FloatTensor(data[np.newaxis, ...]), case["domain"]
