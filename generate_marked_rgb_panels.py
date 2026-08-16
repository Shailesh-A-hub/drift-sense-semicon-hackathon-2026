#!/usr/bin/env python3
"""
Generate marked RGB Optical Visual Inspection Panels for all 60 test pairs.
Features:
- Full-Color Infernal / Optical Metrology Material Grading
- Green Bounding Box for Ground Truth
- Cyan / Gold Bounding Box for Prediction
- Complete side-by-side [100x Reference | 10x Search with Bounding Boxes]
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np


def apply_optical_color(img_u8: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)


def create_marked_rgb_panel(
    ref_u8: np.ndarray,
    search_u8: np.ndarray,
    gt_x: float,
    gt_y: float,
    pred_x: float,
    pred_y: float,
    error_px: float,
    pair_id: int,
    arch: str,
    box_size: int = 100,
) -> np.ndarray:
    # 1. Apply false-color optical material map
    ref_rgb = apply_optical_color(ref_u8)
    search_rgb = apply_optical_color(search_u8)

    half_r = box_size // 2

    # 2. Draw Ground Truth Bounding Box in GREEN
    gt_x1, gt_y1 = int(max(0, gt_x - half_r)), int(max(0, gt_y - half_r))
    gt_x2, gt_y2 = int(min(search_rgb.shape[1], gt_x + half_r)), int(min(search_rgb.shape[0], gt_y + half_r))
    cv2.rectangle(search_rgb, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 3)
    cv2.putText(
        search_rgb,
        "Ground Truth",
        (gt_x1, max(25, gt_y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # 3. Draw Predicted Bounding Box in CYAN (Pass) or RED (Shift)
    pred_x1, pred_y1 = int(max(0, pred_x - half_r)), int(max(0, pred_y - half_r))
    pred_x2, pred_y2 = int(min(search_rgb.shape[1], pred_x + half_r)), int(min(search_rgb.shape[0], pred_y + half_r))
    pred_color = (255, 255, 0) if error_px <= 5.0 else (0, 0, 255)
    cv2.rectangle(search_rgb, (pred_x1, pred_y1), (pred_x2, pred_y2), pred_color, 3)
    cv2.putText(
        search_rgb,
        f"Predicted (Err: {error_px:.2f}px)",
        (pred_x1, min(search_rgb.shape[0] - 10, pred_y2 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        pred_color,
        2,
    )

    # Resize search image for panel display
    disp_search = cv2.resize(search_rgb, (600, 600))

    # Reference panel
    disp_ref = cv2.resize(ref_rgb, (300, 300))
    ref_panel = np.zeros((600, 340, 3), dtype=np.uint8)
    ref_panel[150:450, 20:320] = disp_ref

    status_str = "PASS (<=2px)" if error_px <= 2.0 else ("PASS (<=5px)" if error_px <= 5.0 else "PERIODIC SHIFT")
    cv2.putText(ref_panel, f"Pair #{pair_id:03d} ({arch.upper()})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(ref_panel, "[RGB Optical Metrology]", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.putText(ref_panel, "100x Reference (1000x1000)", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(ref_panel, f"Error: {error_px:.2f} px", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
    cv2.putText(ref_panel, f"Status: {status_str}", (20, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)

    divider = np.ones((600, 4, 3), dtype=np.uint8) * 120
    panel = np.hstack([ref_panel, divider, disp_search])
    return panel


def generate_all_marked_rgb_panels(
    png_dir="drift_sense_submission/dataset_png",
    pred_json="drift_sense_submission/predictions.json",
    out_dir="drift_sense_submission/visual_inspections_rgb",
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(pred_json, "r") as f:
        predictions = json.load(f)

    print(f"Generating marked RGB Optical visual inspection panels for {len(predictions)} pairs...")

    for p in predictions:
        pair_id = p["pair_id"]
        arch = p["architecture"]
        gt_x = p["gt_x"]
        gt_y = p["gt_y"]
        pred_x = p["pred_x"]
        pred_y = p["pred_y"]
        error_px = p["error_px"]

        ref_file = Path(png_dir) / f"pair_{pair_id:03d}_ref.png"
        search_file = Path(png_dir) / f"pair_{pair_id:03d}_search.png"

        ref_u8 = cv2.imread(str(ref_file), cv2.IMREAD_GRAYSCALE)
        search_u8 = cv2.imread(str(search_file), cv2.IMREAD_GRAYSCALE)

        if ref_u8 is None or search_u8 is None:
            continue

        panel = create_marked_rgb_panel(ref_u8, search_u8, gt_x, gt_y, pred_x, pred_y, error_px, pair_id, arch)
        save_path = out_path / f"inspection_rgb_{pair_id:03d}.png"
        cv2.imwrite(str(save_path), panel)

    print(f"[Done] Saved marked RGB inspection panels to: {out_path}")


if __name__ == "__main__":
    generate_all_marked_rgb_panels()
