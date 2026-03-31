# VAE-MMD Brain Metastasis Segmentation

## Paper (LaTeX PDF Version)

👉 **[Click here to view the paper (PDF)](./Paper.pdf)**

---
Domain adaptation for multi-center brain metastasis segmentation using a Variational Autoencoder with Maximum Mean Discrepancy (VAE-MMD) regularization. The model aligns latent representations across four heterogeneous MRI datasets before nnU-Net segmentation training.

---

## Overview

Brain metastasis segmentation suffers from severe domain shift across institutions due to differences in scanner hardware, acquisition protocols, and patient populations. This repository implements a VAE-MMD pipeline that:

1. Applies N4 bias field correction to all datasets.
2. Trains a U-Net-style VAE with MMD regularization to align latent space distributions across Stanford, UCSF, UCLM, and PKG datasets.
3. Reconstructs training and validation volumes through the VAE encoder-decoder to produce domain-normalized images.
4. Trains nnU-Net with 5-fold cross-validation on the reconstructed data.
5. Evaluates on the held-out test set.

---

## Repository Structure

```
vae-mmd-brain-metastasis/
├── code/
│   ├── n4_bias_correction.py          # N4 bias field correction for all datasets
│   ├── train_vae_mmd_adapter.py       # VAE-MMD training
│   ├── evaluate_domain_adaptation.py  # t-SNE and domain classifier evaluation
│   ├── prepare_vae_dataset.py         # 6:2:2 split generation
│   ├── reconstruct_with_vae.py        # VAE reconstruction of train/val
│   ├── train_nnunet_cv.py             # 5-fold nnUNet training
│   └── evaluate_nnunet.py             # Test set inference and metrics
└── utils/
    ├── __init__.py
    ├── vae_mmd_model.py               # HighQualityVAE, Discriminator3D
    ├── vae_mmd_losses.py              # MMD, SSIM, composite VAE loss
    ├── vae_mmd_data.py                # FourDatasetPreprocessed dataset class
    ├── vae_mmd_eval.py                # Domain adaptation evaluation utilities
    └── vae_mmd_metrics.py             # Segmentation metrics (HD95, sDice, F1, F2)
```

---

## Datasets

| Dataset | Reference | Access |
|---------|-----------|--------|
| Stanford (BrainMetShare) | Grovik et al., JMRI 2020 | [Link](https://aimi.stanford.edu/brainmetshare) |
| UCSF-BMSR | Rudie et al., Radiology AI 2024 | [Link](https://radiology.ucsf.edu/research/labs/hess#accordion-patient-research-datasets) |
| UCLM | Ocana-Tienda et al., Scientific Data 2023 | [Link](https://doi.org/10.6084/m9.figshare.21709365) |
| PKG (Brain-Mets-Lung) | Chadha et al., TCIA 2025 | [Link](https://doi.org/10.7937/k0sm-y874) |

All datasets must be obtained directly from the respective sources and are not redistributed here.

---

## Installation

**Python >= 3.9** is required.

```bash
git clone https://github.com/YuchenYang/vae-mmd-brain-metastasis.git
cd vae-mmd-brain-metastasis
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nibabel SimpleITK scipy scikit-learn matplotlib seaborn tqdm blosc2
```

Install the specific nnU-Net version used in this work:

```bash
pip install --force-reinstall git+https://github.com/MIC-DKFZ/nnUNet.git@86606c53ef9f556d6f024a304b52a48378453641
```

Set nnU-Net environment variables:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

---

## Pipeline

### Step 1 — N4 Bias Field Correction

```bash
python code/n4_bias_correction.py \
    --stanford_in  /path/to/stanford/raw \
    --stanford_out /path/to/stanford/n4 \
    --ucsf_in      /path/to/ucsf/raw \
    --ucsf_out     /path/to/ucsf/n4 \
    --uclm_in      /path/to/uclm/raw \
    --uclm_out     /path/to/uclm/n4 \
    --pkg_in       /path/to/pkg/raw \
    --pkg_out      /path/to/pkg/n4
```

### Step 2 — nnU-Net Preprocessing (Dataset999)

Organize the corrected images into nnU-Net raw format, then run:

```bash
nnUNetv2_plan_and_preprocess -d 999 -c 3d_fullres --verify_dataset_integrity
```

### Step 3 — Generate Dataset Split

```bash
python code/prepare_vae_dataset.py \
    --preprocessed_dir /path/to/nnUNet_preprocessed/Dataset999_FourDatasets/nnUNetPlans_3d_fullres \
    --output_dir       /path/to/splits \
    --train_ratio 0.6 \
    --val_ratio   0.2 \
    --random_state 42
```

Produces `four_datasets_split.npz` with train/val/test case lists per domain.

### Step 4 — Train VAE-MMD

```bash
python code/train_vae_mmd_adapter.py \
    --preprocessed_dir /path/to/nnUNet_preprocessed/Dataset999_FourDatasets/nnUNetPlans_3d_fullres \
    --output_dir       /path/to/vae_output \
    --latent_dim 512 \
    --batch_size 4 \
    --num_epochs 100 \
    --lr 1e-4
```

Checkpoints are saved every `--save_interval` epochs. Best model (lowest VAE loss) is saved as `vae_mmd_best.pth`.

### Step 5 — Evaluate Domain Adaptation

```bash
python code/evaluate_domain_adaptation.py \
    --checkpoint       /path/to/vae_mmd_best.pth \
    --preprocessed_dir /path/to/nnUNet_preprocessed/Dataset999_FourDatasets/nnUNetPlans_3d_fullres \
    --output_dir       /path/to/domain_eval_output
```

Produces a t-SNE scatter plot (`tsne_domain_alignment.png`) and domain classifier confusion matrices (`confusion_matrix.png`). A lower domain classification accuracy after VAE-MMD indicates better latent space alignment.

### Step 6 — Reconstruct Train and Val with VAE

```bash
python code/reconstruct_with_vae.py \
    --checkpoint        /path/to/vae_mmd_best.pth \
    --preprocessed_dir  /path/to/nnUNet_preprocessed/Dataset999_FourDatasets/nnUNetPlans_3d_fullres \
    --preprocessed_base /path/to/nnUNet_preprocessed/Dataset999_FourDatasets \
    --split_file        /path/to/splits/four_datasets_split.npz \
    --output_base       /path/to/nnUNet_preprocessed/Dataset998_FourDatasets_VAE
```

Only train and val cases are reconstructed. Test cases are intentionally excluded to preserve evaluation integrity.

### Step 7 — Train nnU-Net with 5-Fold Cross-Validation

```bash
python code/train_nnunet_cv.py \
    --preprocessed_base /path/to/nnUNet_preprocessed/Dataset998_FourDatasets_VAE \
    --split_file        /path/to/splits/four_datasets_split.npz \
    --dataset_id        998 \
    --folds 0 1 2 3 4 \
    --nnunet_results    /path/to/nnUNet_results
```

### Step 8 — Evaluate on Test Set

```bash
python code/evaluate_nnunet.py \
    --split_file     /path/to/splits/four_datasets_split.npz \
    --nnunet_results /path/to/nnUNet_results \
    --dataset_id     998 \
    --dataset_name   Dataset998_FourDatasets_VAE \
    --output_dir     /path/to/eval_output \
    --stanford_dir   /path/to/stanford/n4 \
    --ucsf_dir       /path/to/ucsf/n4 \
    --uclm_dir       /path/to/uclm/n4 \
    --pkg_dir        /path/to/pkg/n4 \
    --folds 0 1 2 3 4 \
    --checkpoint checkpoint_best.pth
```

Inference is performed on the original (non-reconstructed) test images. Results are saved as `evaluation_results.pkl` and printed per dataset.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Sensitivity | TP / (TP + FN) |
| Precision | TP / (TP + FP) |
| F1 | Standard Dice coefficient |
| F2 | Recall-weighted F-score (beta=2) |
| sDice | Surface Dice at 1 mm tolerance |
| HD95 | 95th percentile Hausdorff Distance (mm), reported as median across cases |

Domain adaptation quality is assessed via a logistic regression domain classifier trained on raw intensity statistics (7D) versus VAE latent means (512D). Lower post-adaptation classification accuracy indicates better domain mixing.

---

## VAE Architecture

The VAE uses a U-Net-style encoder-decoder with skip connections:

- **Encoder**: 128³ → 64³ → 32³ → 16³ → 8³ (four strided Conv3d stages with ResidualBlock3D)
- **Latent space**: 512-dimensional with reparameterization trick
- **Decoder**: 8³ → 128³ with skip concatenation from each encoder stage
- **Output activation**: tanh (images normalized to [-1, 1])

The composite training loss combines MSE reconstruction (λ=300), L1 reconstruction (λ=150), SSIM (λ=50), KL divergence (λ=0.1), multi-scale RBF MMD across all domain pairs (λ=10), and LSGAN adversarial loss (λ=5).

---

## References

```bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}

@article{gretton2012kernel,
  title={A kernel two-sample test},
  author={Gretton, Arthur and Borgwardt, Karsten M and Rasch, Malte J and Sch{\"o}lkopf, Bernhard and Smola, Alexander},
  journal={The journal of machine learning research},
  volume={13},
  number={1},
  pages={723--773},
  year={2012},
  publisher={JMLR. org}
}

@inproceedings{kingma2014auto,
  title={Auto-encoding variational bayes},
  author={Kingma, Diederik P and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2013}
}

@article{grovik2019deep,
  title={Deep learning enables automatic detection and segmentation of brain metastases on multisequence MRI},
  author={Gr{\o}vik, Endre and Yi, Darvin and Iv, Michael and Tong, Elizabeth and Rubin, Daniel and Zaharchuk, Greg},
  journal={Journal of Magnetic Resonance Imaging},
  volume={51},
  number={1},
  pages={175--182},
  year={2020},
  publisher={Wiley Online Library}
}

@article{rudie2024ucsf,
  title={The University of California San Francisco brain metastases stereotactic radiosurgery (UCSF-BMSR) MRI dataset},
  author={Rudie, Jeffrey D and others},
  journal={Radiology: Artificial Intelligence},
  volume={6},
  number={2},
  pages={e230126},
  year={2024},
  publisher={Radiological Society of North America}
}

@article{ocana2023uclm,
  title={A comprehensive dataset of annotated brain metastasis MR images with clinical and radiomic data},
  author={Oca{\~n}a-Tienda, Beatriz and others},
  journal={Scientific data},
  volume={10},
  number={1},
  pages={208},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

@misc{pkg2023tcia0,
  title={MR Imaging and Segmentations with Matched Brain Biopsy Pathology Slides from Patients with Brain Metastases from Primary Lung Cancer},
  author={Chadha, Seena and others},
  year={2025},
  howpublished={The Cancer Imaging Archive},
  doi={10.7937/k0sm-y874}
}
