# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection
### Applied Materials / SEMICON India Hackathon 2026
**Team:** WaferWise (Vellore Institute of Technology, Vellore)  
**Tagline:** *Finding the right site in a sea of patterns.*  
**Members:** SHAILESH A, SANJAY K, Hamsavarthan H  
**Contact:** `sanjay.k2024d@vitstudent.ac.in` | `+91 7200456930`

---

## 1. Problem Overview
In sub-10nm semiconductor fabrication (DRAM & FinFET nodes), Scanning Electron Microscopes (SEMs) experience mechanical stage drift, thermal hysteresis, and beam jitter during wafer navigation. 
Because semiconductor dies contain dense, repeating periodic patterns, standard template matching fails due to periodic ambiguity (locking onto identical neighboring unit cells).

**The Challenge:** Given a $1000 \times 1000$ high-magnification Reference Image ($100\times$) and a $1000 \times 1000$ wide-field Search Image ($10\times$), accurately locate the target reference pattern ($10:1$ scale reduction) and return its center coordinates $(x, y)$ in search-image pixels.

---

## 2. Key Architecture & Methodology

```text
[100x Ref (1000x1000)]               [10x Search (1000x1000)]
         ↓                                      ↓
[Gaussian Pre-filtering]             [Gaussian Pre-filtering]
         ↓                                      ↓
[10:1 Scale Downsampling]                       |
  (Multi-Scale 9:1 - 11:1)                      |
         ↓                                      |
[Multi-Angle Rotation Search]                   |
  (-15° to +15° in steps)                       |
         ↓                                      ↓
         └───────────→ [Dense ZNCC Map] ←───────┘
                              ↓
                [Top-K Candidate Extraction]
                              ↓
          [Closest-to-Center Decision Rule Gating]
                              ↓
              [Sub-Pixel Coordinate (x, y)]
```

### Key Highlights:
1. **10:1 Cross-Magnification Alignment:** Explicitly downsamples the $100\times$ reference to $10\times$ scale space.
2. **Multi-Angle ZNCC:** Searches orientation variation ($1^\circ - 2^\circ$ jitter up to $\pm 15^\circ$).
3. **Applied Materials Decision Rule:** Resolves periodic ambiguity by selecting high-confidence matches closest to the search center $(500, 500)$.
4. **RGB Optical / Color Grading Extension (Bonus):** Generalizes grayscale SEM micrographs into calibrated multi-spectral optical material maps (Silicon, Oxide, Metal).

---

## 3. Results & Pass Rates

Evaluated on 60+ varied, independently generated DRAM & FinFET test pairs under severe noise:

| Metric | Result |
| :--- | :---: |
| **Median Localization Error** | **$1.10\text{ px}$** |
| **Pass Rate ($\le 5\text{ px}$ Threshold)** | **$95.0\%$** |
| **Pass Rate ($\le 4\text{ px}$ Threshold)** | **$93.3\%$** |
| **Pass Rate ($\le 2\text{ px}$ Threshold)** | **$88.3\%$** |
| **Pass Rate ($\le 1\text{ px}$ Sub-pixel)** | **$78.3\%$** |
| **Average Latency per Pair** | **$0.18\text{ seconds}$** |

---

## 4. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Shailesh-A-hub/drift-sense-semicon-hackathon-2026.git
cd drift-sense-semicon-hackathon-2026

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Usage & Execution Commands

### 1. Run Single Pair Localization
```bash
python localize.py --reference path/to/reference.npy --search path/to/search.npy --output-dir results
```

### 2. Run Batch Evaluation on Test Manifest
```bash
python localize.py --batch drift_sense_submission/predictions.json --output-dir results
```

### 3. Run with Color Grading Extension (Bonus Feature)
```bash
python localize.py --reference path/to/reference.png --search path/to/search.png --color-grade --output-dir results
```

### 4. Convert .NPY Dataset to .PNG Images
```bash
python convert_npy_to_png.py --input-dir dataset_npy/ --output-dir dataset_png/
```

---

## 6. Repository File Structure

```text
drift-sense-semicon-hackathon-2026/
├── README.md                                  # Complete documentation & commands
├── requirements.txt                           # Python dependencies
├── localize.py                                # Main localization & evaluation engine
├── drift_sense_dataset_generator.py           # Physics-grounded SEM generator
├── convert_npy_to_png.py                      # Visual inspection & conversion module
├── metadata.json                              # Ground truth physics parameter specs
├── drift_sense_submission.zip                 # Complete submission package
├── results/                                   # Evaluation logs & color-graded images
└── references/                                # Academic citations & SEM handbooks
```

---

## 7. References & Academic Citations
1. **JEOL SEM A-to-Z Handbook**: Principles of Secondary Electron Yield and Edge-Brightening Effects in Semiconductor SEM Inspection.
2. **J. P. Lewis**, *"Fast Normalized Cross-Correlation,"* Vision Interface, 1995.
3. **Zhai et al.**, *"A comprehensive review of deep learning-based real-world image restoration,"* *IEEE Access*, vol. 11, pp. 21049–21067, 2023.
4. **V. Monga et al.**, *"Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing,"* *IEEE Signal Processing Magazine*, vol. 38, no. 2, pp. 18–44, 2021.
5. **Applied Materials / SEMICON India 2026**: Problem Statement 1 Guidelines on Navigation-Error Recovery.
