# IR Spectroscopy + Machine Learning: Automated Chemical Classification

> Classifying unknown benzene derivatives from IR spectra using PCA dimensionality reduction and k-Nearest Neighbours — built from scratch in a physical chemistry lab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/ir-chem-ml/blob/main/notebooks/7a_Machine_Learning_Part_I.ipynb)

---

## What This Project Does

This project builds an end-to-end ML pipeline that can look at an infrared (IR) spectrum of an unknown substituted benzene compound and automatically classify it as **ortho-**, **meta-**, or **para-** substituted — purely from the spectral fingerprint.

**The real-world motivation:** Chemists routinely need to distinguish structural isomers. IR spectroscopy is fast and cheap, but reading spectra by eye is slow and error-prone. This pipeline automates the classification.

**The pipeline, in order:**
1. Raw IR spectra are collected on a Shimadzu FTIR spectrometer (400–4000 cm⁻¹)
2. Data is cleaned: interpolated to integer wavenumbers, normalised by area, and narrowed to the fingerprint region (630–880 cm⁻¹)
3. PCA reduces 251 wavenumber features → 20 principal components
4. A k-NN classifier (k=3) is trained on 76 compounds (380 spectra total, 5 repeats each)
5. The trained model classifies new, unseen spectra collected in the lab

**Result:** Successful automated classification of two unknown aminobenzoic acid isomers.

---

## Repository Structure

```
ir-chem-ml/
│
├── notebooks/                    # Jupyter notebooks (the full learning + analysis journey)
│   ├── 0_Jupyter_Notebook.ipynb          # Environment setup
│   ├── 1a_Introductory_Python.ipynb      # Python fundamentals
│   ├── 2_More_Python.ipynb               # Loops, lists, data structures
│   ├── 3_Functions__Objects_and_Libraries.ipynb  # Functions + numerical integration
│   ├── 4_Pandas_Introduction.ipynb       # DataFrames + first IR data import
│   ├── 5a_Data_Pre_Processing_Functions.ipynb    # Building interpolation + normalisation
│   ├── 5b_Data_Pre_Processing_Implementation.ipynb  # Bulk loading 380 spectra
│   ├── 6_Data_Pre_Processing_Principal_Component_Analysis.ipynb  # PCA + 3D visualisation
│   ├── 7a_Machine_Learning_Part_I.ipynb  # Train/test split + k-NN model
│   └── 7b_Machine_Learning_Part_II.ipynb # Classifying personally collected unknowns
│
├── src/
│   └── C317.py                   # Core library: all pre-processing + PCA functions
│
├── data/
│   ├── raw_ir_spectra/           # ← PUT YOUR 380 .txt TRAINING SPECTRA HERE
│   │                             #   (p-xylene_1.txt ... p-xylylenediamine_5.txt etc.)
│   └── your_collected_spectra/   # Personally collected unknowns (A_1.txt – A_5.txt)
│       ├── A_1.txt
│       ├── A_2.txt
│       ├── A_3.txt
│       ├── A_4.txt
│       └── A_5.txt
│
├── results/                      # Saved plots, model outputs (add your own)
├── docs/                         # Extra documentation / report
├── requirements.txt
└── README.md
```

---

## Quickstart

### Run in Google Colab (no setup needed)
Click the badge at the top, or open any notebook directly in Colab.

You'll need to mount your Google Drive and point the paths in `C317.py` at your data folder:
```python
drive.mount('/content/drive')
# Then update this path in C317.py:
os.scandir("/content/drive/MyDrive/Raw_IR_Spectra")
```

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/ir-chem-ml.git
cd ir-chem-ml
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## The Core Library (`src/C317.py`)

All pre-processing logic lives here, built incrementally across notebooks 3–6:

| Function | What it does |
|---|---|
| `interpolation(df)` | Converts decimal wavenumber indices → integer wavenumbers via linear interpolation |
| `normalisation(df)` | Divides each spectrum by its area (trapezium rule), making spectra comparable |
| `narrow_range(df)` | Restricts analysis to the fingerprint region: 630–880 cm⁻¹ |
| `load_spectra(n)` | Reads all 380 `.txt` spectra, pre-processes each, and returns a combined DataFrame. If `n > 0`, also applies PCA to `n` components |
| `perform_pca(n)` | Runs sklearn PCA and returns a DataFrame of principal components with compound labels |

---

## Data Format

Raw spectra are exported from the Shimadzu FTIR software as `.txt` files with this structure:

```
##TITLE=No Description
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=%T
400.232577    53.327514
400.714785    79.298138
...
```

The first 4 header rows are skipped on import. Files are named `compound_N.txt` where `N` is the repeat number (1–5).

**Training data (380 files):** compounds include p-xylene, p-xylylenediamine, p-trans-anethole, and 73 others — all mono-substituted benzene derivatives labelled o/m/p.

**Test data (your unknowns):** `A_1.txt` through `A_5.txt` — 5 repeats of an unknown aminobenzoic acid isomer collected personally in the lab.

---

## Key Results

- **PCA 3D plot** clearly separates *ortho*, *meta*, and *para* compound clusters in principal component space — validating that the fingerprint region (630–880 cm⁻¹) captures substitution pattern information
- **k-NN classifier (k=3, 20 PCA components, 67:33 train/test split)** successfully classifies held-out test compounds
- **Unknown spectra (A_1–A_5)** classified correctly against the demonstrator's ground truth

---

## What I Learned / Built

This was a complete project from first principles:

- Wrote numerical integration (trapezium rule) from scratch in Python — then used it for normalisation
- Built a custom data loader that handles 380 files automatically using `os.scandir()`
- Understood *why* PCA works geometrically before applying `sklearn` — not just calling a black box
- Designed the train/test split carefully: splitting by *compound* (not by spectrum) to avoid data leakage from the 5 repeats
- Applied a real ML model to data I personally collected on lab equipment

---

## Stack

- Python 3 · NumPy · Pandas · scikit-learn · Matplotlib
- Google Colab (development environment)
- Shimadzu FTIR spectrometer (data collection)

---

## About

This was a third-year physical chemistry computing lab (C317) at the University of Oxford. The full exercise spans 8 notebooks and was designed to teach scientific computing and ML from the ground up, applied to real spectroscopic data.

---

## Where to Put Your Data

If you want to reproduce the full results:

1. **Training spectra (380 files):** Place all `.txt` files from the original spectral library into `data/raw_ir_spectra/`. File names should follow the pattern `compoundname_N.txt` (e.g. `p-xylene_1.txt`). These are the files visible in the screenshot — p-xylylenediamine, p-xylene, p-trans-anethole, etc.

2. **Your collected unknowns:** Already included in `data/your_collected_spectra/` as `A_1.txt` through `A_5.txt`.

3. **Update the path in `src/C317.py`:** Change the `os.scandir()` path to point to `data/raw_ir_spectra/` (or your Google Drive equivalent).
