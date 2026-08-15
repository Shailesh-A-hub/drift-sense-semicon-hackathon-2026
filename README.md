# Drift-Sense: Semiconductor Image Drift Estimation

This repository contains a CPU-compatible Dense Zero-Mean Normalized Cross-Correlation (ZNCC) baseline for estimating 2D translation drift between a reference semiconductor image and a shifted query image.

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
