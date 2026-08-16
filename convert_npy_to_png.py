#!/usr/bin/env python3
"""
NPY to PNG Image Conversion & Visualization Module
SEMICON AI Hackathon - Image Restoration Task

This module provides standalone CLI and programmatic utilities to convert .npy 
restoration outputs and dataset arrays into .png images for visual inspection,
evaluation reports, and side-by-side comparisons.

Key Features:
- Single-file and batch directory conversion (.npy -> .png)
- Standardized range normalization:
    * 'clip_01': Clips float array to [0.0, 1.0] and scales to [0, 255] (uint8) or [0, 65535] (uint16).
    * 'minmax': Dynamic min-max stretching to [0, 255] (ideal for noisy inputs with negative/out-of-bound values).
    * 'direct': Direct conversion for arrays already in [0, 255] range.
- Dual submission export helper: Saves both .npy (for metric scoring) and .png (for visual review).
- Side-by-side comparison generator: Creates visual panels [NoisyLR | Restored | GroundTruth].

Usage:
    # 1. Convert an entire directory of .npy predictions to .png:
    python convert_npy_to_png.py --input-dir results/npy/ --output-dir results/png/

    # 2. Convert a single .npy file:
    python convert_npy_to_png.py --input-file results/003200.npy --output-file results/003200.png

    # 3. Generate side-by-side comparison panels:
    python convert_npy_to_png.py --compare --lr-dir data/Test_NoisyLR --restored-dir results/npy --output-dir comparisons/
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


def normalize_to_uint8(
    arr: np.ndarray,
    mode: str = "clip_01",
) -> np.ndarray:
    """
    Normalizes a NumPy array to an 8-bit unsigned integer image (uint8: 0 - 255).

    Args:
        arr: Input numpy array (2D or 3D).
        mode: Normalization strategy:
            - 'clip_01': Clips values to [0.0, 1.0] and scales to [0, 255]. (Recommended for GT & Restored outputs)
            - 'minmax': Linearly stretches [min, max] to [0, 255]. (Useful for NoisyLR with out-of-range values)
            - 'direct': Assumes values are already [0, 255], clips and casts to uint8.

    Returns:
        np.ndarray: uint8 image array formatted for OpenCV/PIL saving.
    """
    arr = arr.squeeze()
    
    # Handle channel-first (C, H, W) -> (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr.squeeze(2)

    arr_f = arr.astype(np.float32)

    if mode == "clip_01":
        clipped = np.clip(arr_f, 0.0, 1.0)
        img_uint8 = np.round(clipped * 255.0).astype(np.uint8)
    elif mode == "minmax":
        min_v, max_v = float(np.min(arr_f)), float(np.max(arr_f))
        if max_v - min_v < 1e-7:
            img_uint8 = np.zeros_like(arr_f, dtype=np.uint8)
        else:
            stretched = (arr_f - min_v) / (max_v - min_v)
            img_uint8 = np.round(stretched * 255.0).astype(np.uint8)
    elif mode == "direct":
        img_uint8 = np.clip(np.round(arr_f), 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported normalization mode: '{mode}'. Choose 'clip_01', 'minmax', or 'direct'.")

    return img_uint8


def normalize_to_uint16(
    arr: np.ndarray,
    mode: str = "clip_01",
) -> np.ndarray:
    """
    Normalizes a NumPy array to a 16-bit unsigned integer image (uint16: 0 - 65535).
    Preserves higher dynamic range for scientific and lossless visual inspection.
    """
    arr = arr.squeeze()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr.squeeze(2)

    arr_f = arr.astype(np.float32)

    if mode == "clip_01":
        clipped = np.clip(arr_f, 0.0, 1.0)
        img_uint16 = np.round(clipped * 65535.0).astype(np.uint16)
    elif mode == "minmax":
        min_v, max_v = float(np.min(arr_f)), float(np.max(arr_f))
        if max_v - min_v < 1e-7:
            img_uint16 = np.zeros_like(arr_f, dtype=np.uint16)
        else:
            stretched = (arr_f - min_v) / (max_v - min_v)
            img_uint16 = np.round(stretched * 65535.0).astype(np.uint16)
    elif mode == "direct":
        img_uint16 = np.clip(np.round(arr_f), 0, 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported normalization mode: '{mode}'.")

    return img_uint16


def convert_npy_file_to_png(
    npy_path: Union[str, Path],
    png_path: Optional[Union[str, Path]] = None,
    mode: str = "clip_01",
    bit_depth: int = 8,
) -> Path:
    """
    Converts a single .npy file to a .png file.

    Args:
        npy_path: Path to the input .npy file.
        png_path: Path to the output .png file (defaults to replacing .npy with .png).
        mode: Normalization mode ('clip_01', 'minmax', 'direct').
        bit_depth: Bit depth for output PNG (8 or 16).

    Returns:
        Path: Output PNG file path.
    """
    npy_path = Path(npy_path)
    if not npy_path.is_file():
        raise FileNotFoundError(f"Input .npy file not found: {npy_path}")

    if png_path is None:
        png_path = npy_path.with_suffix(".png")
    else:
        png_path = Path(png_path)

    png_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.load(npy_path)
    if bit_depth == 16:
        img = normalize_to_uint16(arr, mode=mode)
    else:
        img = normalize_to_uint8(arr, mode=mode)

    cv2.imwrite(str(png_path), img)
    return png_path


def convert_npy_directory_to_png(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.npy",
    mode: str = "clip_01",
    bit_depth: int = 8,
    max_workers: int = 8,
) -> int:
    """
    Batch converts all .npy files in a directory to .png.

    Args:
        input_dir: Source directory containing .npy files.
        output_dir: Destination directory for .png files (defaults to input_dir).
        pattern: Glob pattern to filter .npy files.
        mode: Normalization mode ('clip_01', 'minmax', 'direct').
        bit_depth: Bit depth (8 or 16).
        max_workers: Number of parallel threads for fast conversion.

    Returns:
        int: Number of converted files.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(list(input_dir.glob(pattern)))
    if not npy_files:
        print(f"[Warning] No files matching '{pattern}' found in {input_dir}")
        return 0

    print(f"Converting {len(npy_files)} files from {input_dir} -> {output_dir} (mode={mode}, {bit_depth}-bit)...")

    def _convert(file_path: Path):
        target_path = output_dir / f"{file_path.stem}.png"
        convert_npy_file_to_png(file_path, target_path, mode=mode, bit_depth=bit_depth)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_convert, npy_files))

    print(f"[Success] Converted {len(npy_files)} files successfully.")
    return len(npy_files)


def save_restored_output(
    restored_array: np.ndarray,
    output_base_dir: Union[str, Path],
    sample_name: str,
    save_npy: bool = True,
    save_png: bool = True,
    png_mode: str = "clip_01",
    bit_depth: int = 8,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Workflow helper to simultaneously save model predictions in both .npy and .png formats.
    Organizes results cleanly into:
        <output_base_dir>/npy/<sample_name>.npy
        <output_base_dir>/png/<sample_name>.png

    Args:
        restored_array: 2D or 3D numpy array from restoration model (float32).
        output_base_dir: Base directory for outputs.
        sample_name: Identifier for the sample (e.g. '000130' or 'sample_001').
        save_npy: Whether to save .npy format.
        save_png: Whether to save .png format.
        png_mode: Normalization mode for PNG ('clip_01', 'minmax').
        bit_depth: PNG bit depth (8 or 16).

    Returns:
        Tuple[Optional[Path], Optional[Path]]: (npy_path, png_path)
    """
    output_base = Path(output_base_dir)
    stem = Path(sample_name).stem
    npy_path = None
    png_path = None

    if save_npy:
        npy_dir = output_base / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        npy_path = npy_dir / f"{stem}.npy"
        np.save(npy_path, restored_array.astype(np.float32))

    if save_png:
        png_dir = output_base / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        png_path = png_dir / f"{stem}.png"
        if bit_depth == 16:
            img = normalize_to_uint16(restored_array, mode=png_mode)
        else:
            img = normalize_to_uint8(restored_array, mode=png_mode)
        cv2.imwrite(str(png_path), img)

    return npy_path, png_path


def create_comparison_panel(
    lr_input: np.ndarray,
    restored: np.ndarray,
    gt: Optional[np.ndarray] = None,
    target_height: int = 256,
) -> np.ndarray:
    """
    Creates a visual side-by-side comparison panel:
    [ Noisy Low-Res (Upscaled) | Model Restored Output | Ground Truth (if available) ]

    Includes text labels for clear evaluator inspection.
    """
    # Convert all inputs to uint8
    lr_u8 = normalize_to_uint8(lr_input, mode="minmax")
    rest_u8 = normalize_to_uint8(restored, mode="clip_01")
    
    # Upscale LR to target height using nearest or bicubic for fair visual comparison
    if lr_u8.shape[0] != target_height:
        lr_vis = cv2.resize(lr_u8, (target_height, target_height), interpolation=cv2.INTER_NEAREST)
    else:
        lr_vis = lr_u8

    if rest_u8.shape[0] != target_height:
        rest_vis = cv2.resize(rest_u8, (target_height, target_height), interpolation=cv2.INTER_NEAREST)
    else:
        rest_vis = rest_u8

    # Convert grayscale to BGR for annotations
    if lr_vis.ndim == 2:
        lr_vis = cv2.cvtColor(lr_vis, cv2.COLOR_GRAY2BGR)
    if rest_vis.ndim == 2:
        rest_vis = cv2.cvtColor(rest_vis, cv2.COLOR_GRAY2BGR)

    # Annotate panels
    cv2.putText(lr_vis, "Noisy LR Input", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(rest_vis, "Restored Prediction", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    panels = [lr_vis, rest_vis]

    if gt is not None:
        gt_u8 = normalize_to_uint8(gt, mode="clip_01")
        if gt_u8.shape[0] != target_height:
            gt_vis = cv2.resize(gt_u8, (target_height, target_height), interpolation=cv2.INTER_NEAREST)
        else:
            gt_vis = gt_u8
        if gt_vis.ndim == 2:
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
        cv2.putText(gt_vis, "Ground Truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        panels.append(gt_vis)

    # Concatenate horizontally with a separator line
    separator = np.ones((target_height, 4, 3), dtype=np.uint8) * 128
    result = []
    for i, p in enumerate(panels):
        result.append(p)
        if i < len(panels) - 1:
            result.append(separator)

    return np.hstack(result)


def main():
    parser = argparse.ArgumentParser(
        description="Convert semiconductor image restoration .npy arrays to .png images for visual inspection."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", type=str, help="Path to a single .npy file to convert.")
    group.add_argument("--input-dir", type=str, help="Path to directory containing .npy files.")
    group.add_argument("--compare", action="store_true", help="Generate side-by-side comparison panels.")

    parser.add_argument("--output-file", type=str, default=None, help="Output .png path (for single file mode).")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for .png images.")
    parser.add_argument("--pattern", type=str, default="*.npy", help="Glob pattern for batch directory mode.")
    parser.add_argument(
        "--mode",
        choices=["clip_01", "minmax", "direct"],
        default="clip_01",
        help="Normalization strategy: 'clip_01' (default for predictions/GT), 'minmax' (for noisy inputs), 'direct'.",
    )
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=8, help="PNG bit depth (8 or 16).")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers for batch mode.")

    # Arguments for --compare mode
    parser.add_argument("--lr-dir", type=str, help="Directory containing NoisyLR .npy files (for --compare mode).")
    parser.add_argument("--restored-dir", type=str, help="Directory containing restored .npy files (for --compare mode).")
    parser.add_argument("--gt-dir", type=str, default=None, help="Directory containing GT .npy files (optional).")
    parser.add_argument("--max-comparisons", type=int, default=20, help="Maximum number of comparison images to create.")

    args = parser.parse_args()

    if args.compare:
        if not args.lr_dir or not args.restored_dir:
            print("Error: --compare requires both --lr-dir and --restored-dir")
            sys.exit(1)
        
        lr_dir = Path(args.lr_dir)
        restored_dir = Path(args.restored_dir)
        out_dir = Path(args.output_dir or "comparisons")
        out_dir.mkdir(parents=True, exist_ok=True)
        gt_dir = Path(args.gt_dir) if args.gt_dir else None

        restored_files = sorted(list(restored_dir.glob("*.npy")))[: args.max_comparisons]
        print(f"Generating {len(restored_files)} comparison panels in {out_dir}...")

        count = 0
        for r_path in restored_files:
            stem = r_path.stem
            lr_path = lr_dir / f"{stem}.npy"
            if not lr_path.exists():
                # Try finding without leading zeros or matching names
                candidates = list(lr_dir.glob(f"*{stem}*.npy"))
                if candidates:
                    lr_path = candidates[0]
                else:
                    continue

            lr_arr = np.load(lr_path)
            rest_arr = np.load(r_path)
            gt_arr = None
            if gt_dir:
                gt_path = gt_dir / f"{stem}.npy"
                if gt_path.exists():
                    gt_arr = np.load(gt_path)

            panel = create_comparison_panel(lr_arr, rest_arr, gt_arr)
            out_img_path = out_dir / f"compare_{stem}.png"
            cv2.imwrite(str(out_img_path), panel)
            count += 1

        print(f"[Success] Generated {count} comparison panels in {out_dir}")

    elif args.input_file:
        out = convert_npy_file_to_png(
            args.input_file,
            args.output_file,
            mode=args.mode,
            bit_depth=args.bit_depth,
        )
        print(f"[Success] Converted: {args.input_file} -> {out}")

    elif args.input_dir:
        convert_npy_directory_to_png(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            mode=args.mode,
            bit_depth=args.bit_depth,
            max_workers=args.workers,
        )


if __name__ == "__main__":
    main()
