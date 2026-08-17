#!/usr/bin/env python3
"""
Regenerate ALL submission assets with the updated SMART-SEM engine.
- 60 test pairs (dataset_png + dataset_npy + dataset_rgb_optical)
- predictions.json / predictions.csv
- 60 grayscale visual inspection panels
- 60 RGB optical visual inspection panels
- slide6 SUCCESS and FAILURE case images
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from smart_sem_engine import smart_sem_localize, gen_pair, add_strong_anchors, gradient_magnitude
from drift_sense_dataset_generator import (
    generate_dram_layout, generate_finfet_layout,
    rotate_scale, apply_edge_brightening, apply_blur,
    apply_poisson_gaussian_noise,
)


OUT = Path("drift_sense_submission")
PNG_DIR = OUT / "dataset_png"
NPY_DIR = OUT / "dataset_npy"
RGB_DIR = OUT / "dataset_rgb_optical"
VIS_DIR = OUT / "visual_inspections"
VIS_RGB_DIR = OUT / "visual_inspections_rgb"

for d in [PNG_DIR, NPY_DIR, RGB_DIR, VIS_DIR, VIS_RGB_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_as_png(arr, path):
    img_u8 = np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img_u8)


def apply_optical_color(img_u8):
    return cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)


def make_inspection_panel(ref, search, gt_x, gt_y, pred_x, pred_y, err, pair_id, arch, rgb=False):
    """Create side-by-side inspection panel with bounding boxes."""
    box = 100
    half = box // 2

    if rgb:
        ref_u8 = np.round(np.clip(ref, 0, 1) * 255).astype(np.uint8)
        search_u8 = np.round(np.clip(search, 0, 1) * 255).astype(np.uint8)
        ref_vis = apply_optical_color(ref_u8)
        search_vis = apply_optical_color(search_u8)
    else:
        ref_vis = cv2.cvtColor(np.round(np.clip(ref, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        search_vis = cv2.cvtColor(np.round(np.clip(search, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # GT box (green)
    gx1, gy1 = int(max(0, gt_x - half)), int(max(0, gt_y - half))
    gx2, gy2 = int(min(search_vis.shape[1], gt_x + half)), int(min(search_vis.shape[0], gt_y + half))
    cv2.rectangle(search_vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
    cv2.putText(search_vis, "Ground Truth", (gx1, max(25, gy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Pred box (cyan if pass, red if fail)
    px1, py1 = int(max(0, pred_x - half)), int(max(0, pred_y - half))
    px2, py2 = int(min(search_vis.shape[1], pred_x + half)), int(min(search_vis.shape[0], pred_y + half))
    color = (255, 255, 0) if err <= 5.0 else (0, 0, 255)
    cv2.rectangle(search_vis, (px1, py1), (px2, py2), color, 3)
    cv2.putText(search_vis, f"Predicted ({err:.2f}px)", (px1, min(search_vis.shape[0] - 10, py2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    disp_search = cv2.resize(search_vis, (600, 600))
    disp_ref = cv2.resize(ref_vis, (300, 300))

    info = np.zeros((600, 340, 3), dtype=np.uint8)
    info[150:450, 20:320] = disp_ref

    mode = "RGB Optical" if rgb else "Grayscale SEM"
    status = "PASS (<1px)" if err <= 1.0 else ("PASS (<5px)" if err <= 5.0 else "SHIFT")

    cv2.putText(info, f"Pair #{pair_id:03d} ({arch.upper()})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info, f"[{mode}]", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.putText(info, "100x Reference", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(info, f"Error: {err:.2f} px", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(info, f"Status: {status}", (20, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    divider = np.ones((600, 4, 3), dtype=np.uint8) * 120
    return np.hstack([info, divider, disp_search])


def main():
    print("=" * 60)
    print("  REGENERATING ALL SUBMISSION ASSETS")
    print("=" * 60)

    predictions = []
    all_errors = []

    # Generate 60 test pairs (same seeds as benchmark)
    for i in range(60):
        if i < 30:
            arch = "dram" if i % 2 == 0 else "finfet"
            seed = 8000 + i
            noise = "standard"
        else:
            arch = "finfet" if (i - 30) % 2 == 0 else "dram"
            seed = 9500 + (i - 30)
            noise = "high_fidelity"

        pair = gen_pair(arch=arch, seed=seed, noise_level=noise)
        ref = pair["reference"]
        search = pair["search"]
        gt_x = pair["true_x"]
        gt_y = pair["true_y"]

        # Run localization
        pred = smart_sem_localize(ref, search)
        pred_x = pred["pred_x"]
        pred_y = pred["pred_y"]
        err = float(np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2))
        all_errors.append(err)

        status = "PASS" if err <= 5.0 else "FAIL"
        print(f"  [{i:02d}] {arch:6s} | Err: {err:.2f} px | {status}")

        # Save NPY
        np.save(str(NPY_DIR / f"pair_{i:03d}_ref.npy"), ref.astype(np.float32))
        np.save(str(NPY_DIR / f"pair_{i:03d}_search.npy"), search.astype(np.float32))

        # Save PNG
        save_as_png(ref, PNG_DIR / f"pair_{i:03d}_ref.png")
        save_as_png(search, PNG_DIR / f"pair_{i:03d}_search.png")

        # Save RGB Optical
        ref_u8 = np.round(np.clip(ref, 0, 1) * 255).astype(np.uint8)
        search_u8 = np.round(np.clip(search, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(RGB_DIR / f"pair_{i:03d}_ref_rgb_optical.png"), apply_optical_color(ref_u8))
        cv2.imwrite(str(RGB_DIR / f"pair_{i:03d}_search_rgb_optical.png"), apply_optical_color(search_u8))

        # Grayscale inspection panel
        panel_gs = make_inspection_panel(ref, search, gt_x, gt_y, pred_x, pred_y, err, i, arch, rgb=False)
        cv2.imwrite(str(VIS_DIR / f"inspection_{i:03d}.png"), panel_gs)

        # RGB inspection panel
        panel_rgb = make_inspection_panel(ref, search, gt_x, gt_y, pred_x, pred_y, err, i, arch, rgb=True)
        cv2.imwrite(str(VIS_RGB_DIR / f"inspection_rgb_{i:03d}.png"), panel_rgb)

        predictions.append({
            "pair_id": i,
            "architecture": arch,
            "gt_x": round(gt_x, 2),
            "gt_y": round(gt_y, 2),
            "pred_x": round(pred_x, 3),
            "pred_y": round(pred_y, 3),
            "error_px": round(err, 3),
            "confidence": pred["confidence"],
            "runtime_ms": pred["runtime_ms"],
        })

    # Save predictions
    with open(str(OUT / "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    import csv
    with open(str(OUT / "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=predictions[0].keys())
        w.writeheader()
        w.writerows(predictions)

    # Slide 6 images: best success and worst failure
    errs = np.array(all_errors)
    best_idx = int(np.argmin(errs))
    worst_idx = int(np.argmax(errs))

    # Regenerate best/worst panels as slide6
    for idx, label in [(best_idx, "SUCCESS"), (worst_idx, "FAILURE")]:
        p = predictions[idx]
        if idx < 30:
            arch = "dram" if idx % 2 == 0 else "finfet"
            seed = 8000 + idx
            noise = "standard"
        else:
            arch = "finfet" if (idx-30) % 2 == 0 else "dram"
            seed = 9500 + (idx-30)
            noise = "high_fidelity"
        pair = gen_pair(arch=arch, seed=seed, noise_level=noise)
        panel = make_inspection_panel(
            pair["reference"], pair["search"],
            p["gt_x"], p["gt_y"], p["pred_x"], p["pred_y"],
            p["error_px"], idx, arch, rgb=False
        )
        cv2.imwrite(str(OUT / f"slide6_{label}_case.png"), panel)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY:")
    print(f"  Total pairs       : {len(predictions)}")
    print(f"  Pass@5px          : {(errs<=5).mean()*100:.1f}% ({int((errs<=5).sum())}/{len(errs)})")
    print(f"  Pass@2px          : {(errs<=2).mean()*100:.1f}%")
    print(f"  Pass@1px          : {(errs<=1).mean()*100:.1f}%")
    print(f"  Median Error      : {np.median(errs):.2f} px")
    print(f"  Mean Error        : {np.mean(errs):.2f} px")
    print(f"  Worst Error       : {np.max(errs):.2f} px")
    print(f"  Best case (#{best_idx})  : {errs[best_idx]:.3f} px")
    print(f"  Worst case (#{worst_idx}) : {errs[worst_idx]:.3f} px")
    print(f"\nFiles saved:")
    print(f"  {NPY_DIR}/ : 120 .npy files")
    print(f"  {PNG_DIR}/ : 120 .png files")
    print(f"  {RGB_DIR}/ : 120 .png files")
    print(f"  {VIS_DIR}/ : 60 inspection panels")
    print(f"  {VIS_RGB_DIR}/ : 60 RGB inspection panels")
    print(f"  {OUT}/predictions.json")
    print(f"  {OUT}/predictions.csv")
    print(f"  {OUT}/slide6_SUCCESS_case.png")
    print(f"  {OUT}/slide6_FAILURE_case.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
