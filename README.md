# Drift-Sense: AI-Powered Navigation-Error Recovery
### SEMICON India Hackathon 2026 — Track 1 (Applied Materials)

Solving navigation-error recovery for SEM wafer inspection tools: given a
Reference image (small, high-res target template) and a Search image (large,
lower-res field of view, 10x scale difference), locate the Reference pattern's
center (x, y) inside the Search image — robust to extreme scale disparity,
arbitrary rotation, SEM edge-brightening artifacts, independent sensor noise,
and periodic ambiguity in DRAM/FinFET arrays.

## Problem Summary
- Output: absolute `(x, y)` pixel center of the matched region in Search
  image coordinates. If multiple equally-good matches exist, return the one
  closest to the Search image's geometric center.
- Core challenge: standard template matching (NCC, phase correlation) fails
  on periodic layouts because shifting by one lattice pitch gives a near-
  identical correlation score, causing catastrophic localization failure.

## Repository Structure
```
dataset_generator/
    drift_sense_dataset_generator.py   # synthetic Reference/Search pair generator
model/                                  # (in progress) matching architecture
inference.py                            # (in progress) standalone eval script
slides/                                 # Phase 1 submission deck
requirements.txt
```

## Approach (Six-Component Pipeline)

1. **Lattice normalization (FFT + autocorrelation cross-check)** — estimates
   pitch/orientation from a denoised power spectrum, cross-validated against
   spatial autocorrelation for noise robustness, then resamples the search
   window to cancel scale (up to 10x) and rotation before any matching runs.
2. **Pitch-aware positional encoding** — replaces generic sinusoidal
   positional encodings in a LoFTR-style matcher with one locked to the
   measured lattice pitch, giving self-attention an explicit periodicity prior.
3. **Multi-peak detection + edge-normalized annulus context descriptor** —
   NMS across all near-identical correlation peaks; disambiguates using a
   context embedding computed on an edge-normalized (gradient) representation
   of the non-periodic surround, explicitly excluding the repeating core.
   Falls back to center-proximity tie-break per the official rule when
   candidates remain ambiguous.
4. **Confidence/rejection head** — trained directly against the generator's
   own ambiguity ground truth (periodic-peak density, pitch-confidence),
   producing an honest, interpretable failure signal.
5. **Severity- and noise-curriculum-controlled synthetic generator** —
   procedurally generates DRAM/FinFET layouts with independent per-image
   noise, physically-grounded edge-brightening, blur, and rotation/scale
   jitter; deliberately oversamples high-noise conditions during training
   since the official test set is stated to be noisier than typical training
   data.
6. **Output contract** — literal `(x, y)` absolute pixel coordinate in Search
   image space, matching the evaluation harness exactly.

## Dataset Generator

```bash
python dataset_generator/drift_sense_dataset_generator.py \
    --n_pairs 30 --arch both --severity train --out_dir dataset
```

Produces `dataset/pair_XXX_ref.npy`, `dataset/pair_XXX_search.npy`, and a
`manifest.json` with ground-truth `(true_x, true_y)` and an `ambiguity_label`
per pair, for both DRAM-style and FinFET-style layouts.

### Mandatory generator properties (per official spec)
- Independent sensor noise per image (distinct RNG streams, no shared seed)
- Physically-grounded edge-brightening (distance-transform halo, not a sharp
  edge highlight) — mimics lateral secondary-electron escape near topographic
  edges
- Blur, rotation, and scale degradation
- Search image noisier than reference (train-time), with an explicit `test`
  severity mode that oversamples noise beyond training levels

## Citations (to be expanded in Slide 9)
- Edge-brightening / secondary-electron yield: JEOL SEM A-to-Z; SE yield
  measurement literature.
- Periodic-pattern matching failure of NCC/phase correlation; LoFTR global
  context matching: Sun et al., "LoFTR: Detector-Free Local Feature Matching
  with Transformers," CVPR 2021. Sarlin et al., "SuperGlue," CVPR 2020.
- DRAM/FinFET structural pitch scaling: IRDS/ITRS roadmap references (TBD).

## Team
SEMICON India Hackathon 2026 — Track 1 submission.
