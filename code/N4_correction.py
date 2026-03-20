#!/usr/bin/env python3
"""
N4 Bias Field Correction for all datasets (Stanford, UCSF, UCLM, PKG).
Corrected images are saved to a separate output directory, preserving
the original data. Labels/segmentations are copied without modification.
"""

import os
import shutil
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import affine_transform


def apply_n4(img_data, spacing, n_iterations=None):
    """
    Apply N4 bias field correction to a 3D image volume.

    Args:
        img_data: numpy array of shape (X, Y, Z)
        spacing: voxel spacing tuple (sx, sy, sz)
        n_iterations: list of iterations per resolution level,
                      defaults to [50, 50, 50, 50]

    Returns:
        Bias-corrected numpy array of the same shape
    """
    if n_iterations is None:
        n_iterations = [50, 50, 50, 50]

    sitk_img = sitk.GetImageFromArray(img_data.astype(np.float32).T)
    sitk_img.SetSpacing([float(s) for s in spacing])

    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_iterations)

    corrected = corrector.Execute(sitk_img, mask)
    return sitk.GetArrayFromImage(corrected).T


def process_stanford(input_dir, output_dir, n_iterations=None):
    """
    Process all Stanford cases.
    Image file: t1_gd.nii.gz
    Label file: seg.nii.gz
    """
    cases = [d for d in sorted(os.listdir(input_dir))
             if os.path.isdir(os.path.join(input_dir, d))]

    print(f"Stanford: {len(cases)} cases")

    for case in tqdm(cases, desc="Stanford"):
        case_in = os.path.join(input_dir, case)
        case_out = os.path.join(output_dir, case)
        os.makedirs(case_out, exist_ok=True)

        img_path = os.path.join(case_in, "t1_gd.nii.gz")
        seg_path = os.path.join(case_in, "seg.nii.gz")

        if not os.path.exists(img_path):
            continue

        img_nib = nib.load(img_path)
        img_data = img_nib.get_fdata()
        spacing = img_nib.header.get_zooms()[:3]

        corrected = apply_n4(img_data, spacing, n_iterations)
        nib.save(
            nib.Nifti1Image(corrected.astype(np.float32), img_nib.affine, img_nib.header),
            os.path.join(case_out, "t1_gd.nii.gz")
        )

        if os.path.exists(seg_path):
            shutil.copy(seg_path, os.path.join(case_out, "seg.nii.gz"))


def process_ucsf(input_dir, output_dir, n_iterations=None):
    """
    Process all UCSF cases.
    Image file: {case}_T1post.nii.gz
    Label file: {case}_seg.nii.gz
    """
    cases = [d for d in sorted(os.listdir(input_dir))
             if os.path.isdir(os.path.join(input_dir, d))]

    print(f"UCSF: {len(cases)} cases")

    for case in tqdm(cases, desc="UCSF"):
        case_in = os.path.join(input_dir, case)
        case_out = os.path.join(output_dir, case)
        os.makedirs(case_out, exist_ok=True)

        img_path = os.path.join(case_in, f"{case}_T1post.nii.gz")
        seg_path = os.path.join(case_in, f"{case}_seg.nii.gz")

        if not os.path.exists(img_path):
            continue

        img_nib = nib.load(img_path)
        img_data = img_nib.get_fdata()
        spacing = img_nib.header.get_zooms()[:3]

        corrected = apply_n4(img_data, spacing, n_iterations)
        nib.save(
            nib.Nifti1Image(corrected.astype(np.float32), img_nib.affine, img_nib.header),
            os.path.join(case_out, f"{case}_T1post.nii.gz")
        )

        if os.path.exists(seg_path):
            shutil.copy(seg_path, os.path.join(case_out, f"{case}_seg.nii.gz"))


def process_uclm(input_dir, output_dir, n_iterations=None):
    """
    Process all UCLM cases.
    Image files end with 'img' and have .nii extension.
    Mask files end with 'msk' and have .nii extension.
    """
    patients = [d for d in sorted(os.listdir(input_dir))
                if os.path.isdir(os.path.join(input_dir, d))]

    print(f"UCLM: {len(patients)} patients")

    for patient in tqdm(patients, desc="UCLM"):
        patient_in = os.path.join(input_dir, patient)
        patient_out = os.path.join(output_dir, patient)
        os.makedirs(patient_out, exist_ok=True)

        files = os.listdir(patient_in)
        img_files = sorted([f for f in files if f.endswith('.nii') and 'img' in f.lower()])
        msk_files = sorted([f for f in files if f.endswith('.nii') and 'msk' in f.lower()])

        for img_file in img_files:
            img_path = os.path.join(patient_in, img_file)
            img_nib = nib.load(img_path)
            img_data = img_nib.get_fdata()
            spacing = img_nib.header.get_zooms()[:3]

            corrected = apply_n4(img_data, spacing, n_iterations)
            nib.save(
                nib.Nifti1Image(corrected.astype(np.float32), img_nib.affine, img_nib.header),
                os.path.join(patient_out, img_file)
            )

        for msk_file in msk_files:
            shutil.copy(
                os.path.join(patient_in, msk_file),
                os.path.join(patient_out, msk_file)
            )


def process_pkg(input_dir, output_dir, n_iterations=None):
    """
    Process all PKG cases.
    Image files contain 't1ce_img' in filename.
    Segmentation files contain 'core_seg' in filename.
    If image and segmentation shapes differ, segmentation is aligned
    to image space using affine transform.
    """
    cases = [d for d in sorted(os.listdir(input_dir))
             if os.path.isdir(os.path.join(input_dir, d))]

    print(f"PKG: {len(cases)} cases")

    for case in tqdm(cases, desc="PKG"):
        case_in = os.path.join(input_dir, case)
        case_out = os.path.join(output_dir, case)
        os.makedirs(case_out, exist_ok=True)

        files = os.listdir(case_in)
        t1ce_files = [f for f in files if 't1ce_img' in f]
        seg_files = [f for f in files if 'core_seg' in f]

        if not t1ce_files:
            continue

        img_path = os.path.join(case_in, t1ce_files[0])
        img_nib = nib.load(img_path)
        img_data = img_nib.get_fdata()
        spacing = img_nib.header.get_zooms()[:3]

        corrected = apply_n4(img_data, spacing, n_iterations)
        nib.save(
            nib.Nifti1Image(corrected.astype(np.float32), img_nib.affine, img_nib.header),
            os.path.join(case_out, t1ce_files[0])
        )

        if seg_files:
            seg_path = os.path.join(case_in, seg_files[0])
            seg_nib = nib.load(seg_path)
            seg_data = seg_nib.get_fdata()

            if img_data.shape != seg_data.shape:
                inv_aff = np.linalg.inv(img_nib.affine)
                transform = inv_aff @ seg_nib.affine
                seg_data = affine_transform(
                    seg_data,
                    matrix=transform[:3, :3],
                    offset=transform[:3, 3],
                    output_shape=img_data.shape,
                    order=0
                )
                seg_out = nib.Nifti1Image(
                    (seg_data > 0).astype(np.uint8), img_nib.affine, img_nib.header
                )
            else:
                seg_out = nib.Nifti1Image(
                    (seg_data > 0).astype(np.uint8), seg_nib.affine, seg_nib.header
                )

            nib.save(seg_out, os.path.join(case_out, seg_files[0]))


def main():
    parser = argparse.ArgumentParser(
        description="Apply N4 bias field correction to all datasets."
    )
    parser.add_argument("--stanford_in",  type=str, required=True,
                        help="Path to Stanford dataset root directory")
    parser.add_argument("--stanford_out", type=str, required=True,
                        help="Output path for corrected Stanford data")
    parser.add_argument("--ucsf_in",      type=str, required=True,
                        help="Path to UCSF dataset root directory")
    parser.add_argument("--ucsf_out",     type=str, required=True,
                        help="Output path for corrected UCSF data")
    parser.add_argument("--uclm_in",      type=str, required=True,
                        help="Path to UCLM dataset root directory")
    parser.add_argument("--uclm_out",     type=str, required=True,
                        help="Output path for corrected UCLM data")
    parser.add_argument("--pkg_in",       type=str, required=True,
                        help="Path to PKG dataset root directory")
    parser.add_argument("--pkg_out",      type=str, required=True,
                        help="Output path for corrected PKG data")
    parser.add_argument("--n_iterations", type=int, nargs="+",
                        default=[50, 50, 50, 50],
                        help="N4 iterations per resolution level (default: 50 50 50 50)")
    args = parser.parse_args()

    for out_dir in [args.stanford_out, args.ucsf_out, args.uclm_out, args.pkg_out]:
        os.makedirs(out_dir, exist_ok=True)

    process_stanford(args.stanford_in,  args.stanford_out, args.n_iterations)
    process_ucsf(args.ucsf_in,          args.ucsf_out,     args.n_iterations)
    process_uclm(args.uclm_in,          args.uclm_out,     args.n_iterations)
    process_pkg(args.pkg_in,            args.pkg_out,      args.n_iterations)

    print("N4 bias field correction complete.")


if __name__ == "__main__":
    main()
