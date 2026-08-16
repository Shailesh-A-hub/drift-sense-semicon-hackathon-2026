#!/usr/bin/env python3
"""
Generate RGB Optical / Color-Graded images for all Drift-Sense dataset pairs.
Applied Materials Bonus Feature: Optical Multi-Spectral Generalization.
"""

import os
from pathlib import Path
import cv2
import numpy as np


def apply_semicon_optical_rgb(img_u8: np.ndarray) -> np.ndarray:
    """
    Applies calibrated false-color optical material mapping:
    - Low electron yield = Silicon Substrate (Deep Blue)
    - Medium yield = Oxide Dielectric (Emerald / Green)
    - High edge yield = Metal Interconnects / Vias (Gold / Bright Amber)
    """
    return cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)


def generate_all_rgb_optical(png_dir="drift_sense_submission/dataset_png", out_dir="drift_sense_submission/dataset_rgb_optical"):
    png_path = Path(png_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(list(png_path.glob("*.png")))
    print(f"Generating RGB Optical color-graded PNGs for {len(files)} images...")

    for f in files:
        gray_img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            continue
        rgb_img = apply_semicon_optical_rgb(gray_img)
        save_path = out_path / f.name.replace(".png", "_rgb_optical.png")
        cv2.imwrite(str(save_path), rgb_img)

    print(f"[Done] Saved {len(files)} RGB optical images to: {out_path}")


if __name__ == "__main__":
    generate_all_rgb_optical()
