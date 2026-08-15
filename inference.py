#!/usr/bin/env python3
"""
Dense ZNCC drift estimator.

Usage:
    python inference.py --reference path/to/reference.png \
                        --query path/to/query.png \
                        --output prediction.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image.astype(np.float32)


def preprocess(image: np.ndarray) -> np.ndarray:
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def zncc_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a = a - a.mean()
    b = b - b.mean()

    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-12:
        return -1.0

    return float(np.sum(a * b) / denom)


def overlap_views(
    reference: np.ndarray,
    query: np.ndarray,
    dx: int,
    dy: int,
):
    height, width = reference.shape

    x0 = max(0, dx)
    x1 = min(width, width + dx)
    y0 = max(0, dy)
    y1 = min(height, height + dy)

    if x1 <= x0 or y1 <= y0:
        return None, None

    ref_patch = reference[y0:y1, x0:x1]
    qry_patch = query[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return ref_patch, qry_patch


def estimate_drift(
    reference: np.ndarray,
    query: np.ndarray,
    max_shift: int = 128,
    min_overlap: int = 64,
):
    if reference.shape != query.shape:
        raise ValueError(
            f"Image sizes must match; got {reference.shape} and {query.shape}"
        )

    reference = preprocess(reference)
    query = preprocess(query)

    best_score = -np.inf
    best_dx, best_dy = 0, 0

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            ref_patch, qry_patch = overlap_views(reference, query, dx, dy)

            if (
                ref_patch is None
                or ref_patch.shape[0] < min_overlap
                or ref_patch.shape[1] < min_overlap
            ):
                continue

            score = zncc_score(ref_patch, qry_patch)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Estimate 2D image drift with dense ZNCC."
    )
    parser.add_argument("--reference", required=True, help="Reference image path")
    parser.add_argument("--query", required=True, help="Shifted/query image path")
    parser.add_argument(
        "--output",
        default="prediction.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=128,
        help="Search range in pixels, in each direction",
    )
    args = parser.parse_args()

    reference = load_gray(args.reference)
    query = load_gray(args.query)

    dx, dy, score = estimate_drift(
        reference,
        query,
        max_shift=args.max_shift,
    )

    result = {
        "pred_dx": int(dx),
        "pred_dy": int(dy),
        "zncc_score": float(score),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
