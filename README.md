# Drift-Sense: Semiconductor Image Drift Estimation

This repository contains a CPU-compatible Dense Zero-Mean Normalized Cross-Correlation (ZNCC) baseline for estimating 2D translation drift between a reference semiconductor image and a shifted query image.

## Current strengths

- **Strong typical-case accuracy:** held-out evaluation achieved a 1.10 px median error, with 88.3% of predictions within 10 px.
- **Interpretable matching:** ZNCC provides a direct correlation score for every predicted alignment, making the decision process inspectable rather than opaque.
- **CPU-compatible deployment:** `inference.py` uses OpenCV and NumPy only; a GPU is not required to run inference.
- **Reproducible workflow:** the repository includes synthetic data generation, a saved evaluation notebook, dependency specification, and a standalone inference entry point.
- **Verified end-to-end operation:** the script was smoke-tested on a controlled translated image pair and returned the expected reverse alignment correction with a ZNCC score of 0.9999945.
- **Documented limitations:** the difference between mean and median error indicates hard outlier cases remain; future work will focus on confidence-based rejection, coarse-to-fine search, and robustness to difficult imaging variations.

## Method

The estimator searches candidate horizontal and vertical translations and selects the alignment with the highest zero-mean normalized cross-correlation score. It returns the correction to apply to the query image so that it aligns with the reference image.

## Repository contents

- `drift_sense_dense_zncc.ipynb` — Colab notebook for data generation, evaluation, and analysis.
- `inference.py` — standalone CPU inference script.
- `dataset_generator/` — synthetic dataset generation utilities.
- `baseline/` and `preprocessing/` — supporting project components.

## Installation

```bash
pip install -r requirements.txt
```

## Run inference

```bash
python inference.py \
  --reference path/to/reference.png \
  --query path/to/query.png \
  --output prediction.json \
  --max-shift 128
```

The output is JSON in this form:

```json
{
  "pred_dx": -23,
  "pred_dy": 17,
  "zncc_score": 0.9999945
}
```

`pred_dx` and `pred_dy` are the translation correction to apply to the query image to align it with the reference image. If an evaluation protocol instead defines labels as the shift originally applied to create the query, reverse the two output signs.

## Validation

Held-out evaluation recorded in the notebook:

- Mean error: 31.00 px
- Median error: 1.10 px
- Within 10 px: 88.3%
- Within 50 px: 95.0%

## CPU smoke test

The standalone script was executed successfully on a synthetic 512 × 512 image pair. The query image was created by applying `(dx, dy) = (+23, -17)` to the reference. The estimator returned the expected reverse alignment correction:

```json
{
  "pred_dx": -23,
  "pred_dy": 17,
  "zncc_score": 0.9999945163726807
}
```

No GPU is required for `inference.py`.
