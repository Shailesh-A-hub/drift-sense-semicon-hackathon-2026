"""
Drift-Sense Track 1 - Synthetic Dataset Generator (v2 - anchor density fix)
Applied Materials / SEMICON India Hackathon 2026

v2 CHANGELOG:
  - Fixed: unique anchor features (defects/particles) are now guaranteed to
    land INSIDE the eventual search crop with a controllable count, instead
    of being scattered across the full canvas where most crops ended up with
    zero anchors (causing every sample to hit maximum ambiguity).
  - Fixed: ambiguity_label now measures actual anchors-in-frame per sample,
    not a static function of pitch/architecture alone.
  - New: --anchor_density flag (sparse/medium/dense) for building a real
    curriculum across difficulty levels.

Generates Reference/Search image pairs for DRAM-style and FinFET-style
periodic semiconductor layouts, with:
  - Independent sensor noise per image (Poisson + Gaussian, no shared seed)
  - Physically-grounded edge-brightening (distance-transform halo model)
  - Gaussian/motion blur
  - Rotation + scale jitter
  - Ground-truth (x, y) center of the reference pattern within the search image
  - Per-sample ambiguity label (based on unique-anchor count in frame) for
    confidence-head supervision
  - Configurable noise severity ('train' vs 'test') so training can be
    deliberately curriculum-shifted toward noisier-than-training test conditions

Usage:
    python drift_sense_dataset_generator.py --n_pairs 30 --arch both \
        --severity train --anchor_density medium --out_dir dataset
"""
import argparse
import json
import os

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, gaussian_filter


def generate_dram_layout(size, pitch=40, line_width=6, seed=None):
    """Periodic word-lines/bit-lines with via dots at intersections."""
    rng = np.random.default_rng(seed)
    layout = np.zeros((size, size), dtype=np.float32)
    for x in range(0, size, pitch):
        layout[:, x:x + line_width] = 1.0
    for y in range(0, size, pitch):
        layout[y:y + line_width, :] = 1.0

    via_r = max(2, line_width // 2)
    yy, xx = np.ogrid[-via_r:via_r + 1, -via_r:via_r + 1]
    via_mask = xx ** 2 + yy ** 2 <= via_r ** 2
    for x in range(pitch // 2, size, pitch):
        for y in range(pitch // 2, size, pitch):
            y0, y1 = max(0, y - via_r), min(size, y + via_r + 1)
            x0, x1 = max(0, x - via_r), min(size, x + via_r + 1)
            sub = layout[y0:y1, x0:x1]
            my, mx = sub.shape
            layout[y0:y1, x0:x1] = np.maximum(sub, via_mask[:my, :mx])
    return layout


def generate_finfet_layout(size, fin_pitch=25, fin_width=4, gate_pitch=180, gate_width=15, seed=None):
    """Dense parallel fins crossed by gate bars."""
    layout = np.zeros((size, size), dtype=np.float32)
    for x in range(0, size, fin_pitch):
        layout[:, x:x + fin_width] = 1.0
    for y in range(0, size, gate_pitch):
        layout[y:y + gate_width, :] = np.maximum(layout[y:y + gate_width, :], 0.8)
    return layout


def add_unique_anchors_in_crop(layout, rng, crop_region, min_anchors=3, max_anchors=8):
    """Injects a controllable number of non-periodic unique features
    (particles/defects/die-boundary-like marks) guaranteed to land inside
    the given crop region. This is what the disambiguation stage (annulus
    context descriptor) keys on to break periodic ambiguity, and what the
    ambiguity_label is now computed from. Returns the number placed."""
    size = layout.shape[0]
    cx0, cy0, cx1, cy1 = crop_region
    cx0, cy0 = max(0, cx0), max(0, cy0)
    cx1, cy1 = min(size, cx1), min(size, cy1)
    n_anchors = rng.integers(min_anchors, max_anchors + 1)
    placed = 0
    for _ in range(n_anchors):
        if cx1 <= cx0 or cy1 <= cy0:
            break
        dx = rng.integers(cx0, cx1)
        dy = rng.integers(cy0, cy1)
        dr = rng.integers(4, 10)
        yy, xx = np.ogrid[-dr:dr + 1, -dr:dr + 1]
        mask = xx ** 2 + yy ** 2 <= dr ** 2
        y0, y1 = max(0, dy - dr), min(size, dy + dr + 1)
        x0, x1 = max(0, dx - dr), min(size, dx + dr + 1)
        sub = layout[y0:y1, x0:x1]
        my, mx = sub.shape
        layout[y0:y1, x0:x1] = np.maximum(sub, 0.5 * mask[:my, :mx])
        placed += 1
    return placed


def apply_edge_brightening(img, halo_width=3, strength=0.6):
    """Physically-grounded SEM edge-brightening: secondary electrons escaping
    laterally near topographic edges create a bright halo of finite width,
    not a sharp line. Modeled via a distance-transform falloff from the
    binary boundary (JEOL SEM A-to-Z; secondary-electron yield literature)."""
    binary = img > 0.5
    dist_out = distance_transform_edt(~binary)
    halo = np.exp(-(dist_out ** 2) / (2 * halo_width ** 2))
    boundary = (dist_out < halo_width) & (dist_out > 0)
    halo_boost = np.zeros_like(img)
    halo_boost[boundary] = strength * halo[boundary]
    return np.clip(img + halo_boost, 0, 1)


def apply_blur(img, sigma):
    """Gaussian blur simulating beam defocus / stage vibration during scan."""
    return gaussian_filter(img, sigma=sigma)


def apply_poisson_gaussian_noise(img, poisson_scale=30, gaussian_sigma=0.03, seed=None):
    """Signal-dependent shot noise (Poisson, from discrete electron arrival)
    plus additive Gaussian noise (detector/amplifier thermal noise).
    IMPORTANT: caller must pass a distinct seed per image so reference and
    search noise realizations are independent (no shared noise)."""
    rng = np.random.default_rng(seed)
    img_scaled = np.clip(img, 0, 1) * poisson_scale
    noisy = rng.poisson(img_scaled).astype(np.float32) / poisson_scale
    noisy += rng.normal(0, gaussian_sigma, img.shape)
    return np.clip(noisy, 0, 1)


def rotate_scale(img, angle, scale, output_size):
    rotated = ndimage.rotate(img, angle, reshape=False, order=1, mode='reflect')
    zoomed = ndimage.zoom(rotated, scale, order=1)
    h, w = zoomed.shape
    if h < output_size or w < output_size:
        pad_h, pad_w = max(0, output_size - h), max(0, output_size - w)
        zoomed = np.pad(zoomed, ((0, pad_h), (0, pad_w)), mode='reflect')
    return zoomed[:output_size, :output_size]


_DENSITY_MAP = {'sparse': (1, 3), 'medium': (3, 6), 'dense': (6, 10)}


def generate_pair(architecture='dram', size=1000, ref_size=100, seed=None,
                   noise_severity='train', anchor_density='medium'):
    """Generates one (Reference, Search) pair with ~10x scale relationship
    and ground-truth (x, y) center location of the reference pattern inside
    the search image (search-image pixel coordinates)."""
    rng = np.random.default_rng(seed)
    full_size = size + 2 * ref_size

    if architecture == 'dram':
        full_layout = generate_dram_layout(full_size, pitch=40, line_width=6, seed=seed)
        pitch = 40
    else:
        full_layout = generate_finfet_layout(full_size, fin_pitch=25, fin_width=4, seed=seed)
        pitch = 25

    gx = rng.integers(ref_size, full_size - ref_size)
    gy = rng.integers(ref_size, full_size - ref_size)
    reference = full_layout[gy:gy + ref_size, gx:gx + ref_size].copy()

    max_off = ref_size
    search_x0 = int(np.clip(gx - rng.integers(0, max_off), 0, full_size - size))
    search_y0 = int(np.clip(gy - rng.integers(0, max_off), 0, full_size - size))

    lo, hi = _DENSITY_MAP[anchor_density]
    n_anchors = add_unique_anchors_in_crop(
        full_layout, rng,
        (search_x0, search_y0, search_x0 + size, search_y0 + size),
        min_anchors=lo, max_anchors=hi
    )

    search_full = full_layout[search_y0:search_y0 + size, search_x0:search_x0 + size].copy()

    true_x = gx - search_x0 + ref_size // 2
    true_y = gy - search_y0 + ref_size // 2

    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.95, 1.05)
    reference = rotate_scale(reference, angle, scale, ref_size)
    reference = apply_edge_brightening(reference, halo_width=2, strength=0.5)
    reference = apply_blur(reference, sigma=rng.uniform(0.3, 0.8))
    reference = apply_poisson_gaussian_noise(reference, poisson_scale=40, gaussian_sigma=0.02, seed=seed)

    search_full = apply_edge_brightening(search_full, halo_width=2, strength=0.5)
    search_full = apply_blur(search_full, sigma=rng.uniform(0.5, 1.2))
    if noise_severity == 'train':
        p_scale, g_sigma = rng.uniform(20, 35), rng.uniform(0.03, 0.06)
    else:
        p_scale, g_sigma = rng.uniform(8, 20), rng.uniform(0.06, 0.12)
    search_full = apply_poisson_gaussian_noise(
        search_full, poisson_scale=p_scale, gaussian_sigma=g_sigma, seed=(seed or 0) + 100000
    )

    # Ambiguity label now reflects actual unique-anchor density in THIS
    # sample's crop, not a static pitch-derived constant. More anchors in
    # frame => easier disambiguation => lower ambiguity.
    ambiguity_label = float(np.clip(1.0 - (n_anchors / 10.0), 0.1, 1.0))

    return {
        'reference': reference, 'search': search_full,
        'true_x': int(true_x), 'true_y': int(true_y),
        'architecture': architecture, 'pitch': pitch,
        'n_anchors_in_frame': n_anchors,
        'ambiguity_label': ambiguity_label,
    }


def generate_dataset(n_pairs, out_dir, architectures=('dram', 'finfet'),
                      noise_severity='train', anchor_density='medium', seed0=0):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i in range(n_pairs):
        arch = architectures[i % len(architectures)] if len(architectures) > 1 else architectures[0]
        pair = generate_pair(architecture=arch, seed=seed0 + i,
                              noise_severity=noise_severity, anchor_density=anchor_density)
        ref_path = os.path.join(out_dir, f'pair_{i:03d}_ref.npy')
        search_path = os.path.join(out_dir, f'pair_{i:03d}_search.npy')
        np.save(ref_path, pair['reference'].astype(np.float32))
        np.save(search_path, pair['search'].astype(np.float32))
        manifest.append({
            'id': i, 'architecture': pair['architecture'],
            'reference_path': ref_path, 'search_path': search_path,
            'true_x': pair['true_x'], 'true_y': pair['true_y'],
            'ambiguity_label': pair['ambiguity_label'],
            'n_anchors_in_frame': pair['n_anchors_in_frame'],
            'pitch': pair['pitch'],
        })
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pairs', type=int, default=30)
    parser.add_argument('--arch', choices=['dram', 'finfet', 'both'], default='both')
    parser.add_argument('--severity', choices=['train', 'test'], default='train')
    parser.add_argument('--anchor_density', choices=['sparse', 'medium', 'dense'], default='medium')
    parser.add_argument('--out_dir', default='dataset')
    parser.add_argument('--seed0', type=int, default=0)
    args = parser.parse_args()

    archs = ('dram', 'finfet') if args.arch == 'both' else (args.arch,)
    manifest = generate_dataset(args.n_pairs, args.out_dir, archs,
                                 args.severity, args.anchor_density, args.seed0)
    print(f"Generated {len(manifest)} pairs in {args.out_dir}/ "
          f"(severity={args.severity}, anchor_density={args.anchor_density})")
    ambiguities = [m['ambiguity_label'] for m in manifest]
    print(f"Ambiguity label range: {min(ambiguities):.2f} - {max(ambiguities):.2f}")
