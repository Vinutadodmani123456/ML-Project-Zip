"""
============================================================
Facial Skin Condition Detection
Phase 2 — Image Preprocessing
============================================================

This module provides the full image preprocessing pipeline:
  - Resize images to a standard size (128 × 128)
  - Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    for illumination normalisation
  - Apply Gaussian blur for mild noise reduction

Functions
---------
load_and_preprocess_image(filepath) → np.ndarray (BGR, 128×128)
preprocess_pipeline(img_bgr)       → np.ndarray (preprocessed BGR)

Run standalone:
    python src/preprocessing.py

Author: MCA Project
"""

import os
import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATASET_DIR   = PROJECT_ROOT / "dataset"

# Target size for all images (height × width)
TARGET_SIZE   = (128, 128)

# CLAHE parameters
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# Gaussian blur parameters (kernel must be odd)
GAUSSIAN_KERNEL     = (3, 3)
GAUSSIAN_SIGMA      = 0          # 0 → OpenCV auto-calculates from kernel


# ── Core Functions ────────────────────────────────────────────────────────────

def preprocess_pipeline(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a BGR image array.

    Steps
    -----
    1. Resize to TARGET_SIZE (128 × 128)
    2. Convert to LAB colour space
    3. Apply CLAHE to the L (lightness) channel
    4. Convert back to BGR
    5. Apply Gaussian blur for noise reduction

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (as returned by cv2.imread).

    Returns
    -------
    np.ndarray
        Preprocessed BGR image of shape (128, 128, 3), dtype uint8.
    """
    # 1. Resize
    img_resized = cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 2. Convert BGR → LAB
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)

    # 3. Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])

    # 4. Convert LAB → BGR
    img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    # 5. Gaussian blur for noise reduction
    img_denoised = cv2.GaussianBlur(img_enhanced, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)

    return img_denoised


def load_and_preprocess_image(filepath: str | Path) -> np.ndarray | None:
    """
    Load an image from disk and apply the full preprocessing pipeline.

    Parameters
    ----------
    filepath : str or Path
        Path to the image file (.jpg / .jpeg / .png / .bmp / .tiff).

    Returns
    -------
    np.ndarray
        Preprocessed BGR image (128, 128, 3), or None if the file cannot be read.
    """
    img = cv2.imread(str(filepath))
    if img is None:
        print(f"  [WARNING] Cannot read image: {filepath}")
        return None
    return preprocess_pipeline(img)


# ── Standalone Demo ───────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Quick smoke-test: preprocesses the first image found in each class folder
    and prints confirmation.
    """
    CLASS_FOLDERS = {
        "Acne":                  "Acne",
        "Rosacea":               "Rosacea",
        "Melasma":               "Melasma",
        "Facial Eczema":         "Facial_Eczema",
        "Seborrheic Dermatitis": "Seborrheic_Dermatitis",
    }
    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    print("\n" + "=" * 60)
    print("  PREPROCESSING — Smoke Test")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print(f"\n[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("        Add images before running this script.")
        sys.exit(1)

    for class_name, folder in CLASS_FOLDERS.items():
        class_dir = DATASET_DIR / folder
        if not class_dir.exists():
            print(f"  [SKIP] Folder not found: {class_dir}")
            continue

        images = [f for f in class_dir.rglob("*")
                  if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]

        if not images:
            print(f"  [SKIP] No images in {class_name}")
            continue

        sample = images[0]
        result = load_and_preprocess_image(sample)
        if result is not None:
            print(f"  [OK] {class_name:<28} → {sample.name}  →  shape: {result.shape}")
        else:
            print(f"  [FAIL] {class_name:<26} → {sample.name}")

    print("\n  Preprocessing pipeline is ready.")
    print("  Run feature_extraction.py to build features/features.csv")
    print("=" * 60)


if __name__ == "__main__":
    _demo()
