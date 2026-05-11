# Hyperspectral Water Stress Detection — Pixel-Level Foundation Layer

**Module status:** Validated capability demonstration · NSCIC Stage 2 supporting evidence
**Module owner:** Aganitha Space Technologies Pvt. Ltd. (CIN: U61309TS2023PTC180401)
**Date:** May 2026

---

## What this module is

A pixel-level hyperspectral classifier that detects water stress in groundnut canopies with **99.96% test accuracy (ROC-AUC 0.9999)**, trained and validated on the IIT Hyderabad TiHAN UC-HSI Crop Variety Dataset. This module is **complementary to the main NSCIC dashboard** in this repository — it operates at a different spatial scale and on a different sensor modality, and demonstrates the lowest-level capability that feeds the multi-scale architecture Anvīkṣaṇa is built around.

## How this complements the NSCIC mandal-level dashboard

The main `dashboard.py` in this repository operates at the **mandal level** across Telangana, using DiCRA satellite-derived NDVI / soil moisture / land surface temperature and NASA POWER reanalysis to produce operational drought intelligence — 487,243 records across 592 mandals, 23 biweekly dates in 2025, validated with spatial cross-validation (ROC-AUC 0.974 ± 0.004).

This hyperspectral module operates at the **pixel level** on individual canopies, using narrow-band reflectance signatures to detect water stress at the level where it physically originates — within the leaf. The two layers describe the same underlying phenomenon (crop water stress) at radically different scales:

| Layer | Spatial unit | Data source | Decision context |
|---|---|---|---|
| Hyperspectral pixel | ~1 m² canopy patch | UAV / proximal HSI sensor | Plant-level diagnosis, precision agriculture |
| **NSCIC mandal model** | ~100 km² administrative unit | DiCRA satellite + NASA POWER | District / state planning, NABARD-grade drought intelligence |
| DGGS aggregation (planned) | Multi-resolution cells | All of the above, fused | National Climate Stack |

The two are designed to plug into the same DGGS-anchored knowledge graph so that on-the-ground sensor measurements can be aggregated upward and regional alerts can be drilled down to ground-truth observations. This is the **multi-scale DGGS-native architecture** that makes Anvīkṣaṇa more than a dashboard.

## Headline result

| Model | Test accuracy | F1 (macro) | ROC-AUC | Test errors |
|---|---|---|---|---|
| Random Forest | 0.9744 | 0.9743 | 0.9976 | 64 / 2,501 |
| SVM (RBF kernel) | 0.9988 | 0.9988 | 1.0000 | 3 / 2,501 |
| **1D-CNN (3 conv blocks)** | **0.9996** | **0.9996** | **0.9999** | **1 / 2,501** |
| Logistic regression, full spectrum (linear) | 0.9992 | — | — | — |
| Logistic regression, single band (band 161 only) | 0.7273 | — | — | — |
| Logistic regression, brightness only | 0.6253 | — | — | — |

**Interpretation.** Three independent classifier families converge at near-perfect separation. Univariate baselines reach only 57–73%, and overall canopy brightness is near chance (62.5%) — so the discriminative signal is genuinely *distributed across the spectrum*, not carried by any single band. The dominant signal lives in bands 158–180 (red-edge / NIR transition), with secondary contributions from the visible green edge (bands 75–90) and the SWIR water-absorption region (bands 250–280). Per-band t-statistics and Random Forest feature importance independently agree on these regions, indicating a real biophysical signature consistent with canonical plant water-stress mechanisms.

See `figures/fig_groundnut_summary.png` for the 4-panel summary.

## Files in this module

| File | Description |
|---|---|
| `README.md` | This document |
| `methodology.md` | Full methodology, results, and caveats writeup |
| `pipeline.py` | Reproducible end-to-end pipeline (load → preprocess → train → evaluate → save) |
| `figures/fig_groundnut_summary.png` | 4-panel composite figure (mean spectra, signal localisation, RF importance, results table) |
| `figures/fig_mean_spectra.png` | Class-conditional mean spectra (±1 std) |
| `figures/fig_difference_spectrum.png` | Difference spectrum and per-band t-statistic |
| `figures/fig_rf_importance.png` | Top-15 informative bands by RF importance |
| `results.json` | Full numerical results, machine-readable |
| `DATA_LICENSING.md` | Data attribution and redistribution policy |

**What is intentionally not included:**

- Raw hyperspectral `.npy` data files (TiHAN's intellectual property — not redistributed)
- Trained model weights (derivative of TiHAN data — available on request from Aganitha Space Technologies, pending TiHAN's redistribution permission)

See `DATA_LICENSING.md` for details on how to obtain the dataset and request model weights.

## Reproducing this work

```bash
# 1. Obtain the IIT-H TiHAN UC-HSI Groundnut Water Stress Dataset
#    from the originator (see DATA_LICENSING.md)
# 2. Place X_GN_31Dec.npy and y_GN_31Dec.npy in a directory
# 3. Run the pipeline
python pipeline.py --data_dir /path/to/data --out_dir ./outputs --epochs 40
```

The script reproduces all metrics, figures, and the trained model. Random seed is fixed at 42.

## Caveats — read before quoting these numbers

This is a **preliminary, internally-validated result**, not yet an operationally-deployed claim. Three known limitations:

1. **Random pixel-level split.** Train and test pixels may originate from spatially adjacent locations within the same plant or canopy. The current result therefore demonstrates *spectral separability* of stress classes, not generalisation to unseen plants or acquisition dates. Leave-one-plant-out cross-validation is the planned next step (pending plant-ID metadata from TiHAN).

2. **Single-site, single-cultivar dataset.** Generalisation across cultivars, soil types, growth stages, and illumination conditions has not yet been evaluated.

3. **Binary stress label.** The current label set is binary (stressed / well-watered). Operational deployment will require continuous severity grading.

These caveats are stated openly in `methodology.md` §6 and in any public communication of these results.

## How this connects to NSCIC and Anvīkṣaṇa modules

- **Arunam** (climate-risk module): Hyperspectral pixel-level stress detections aggregate via DGGS indexing into regional stress indices, fused with satellite-derived soil moisture and rainfall anomalies. The NSCIC dashboard at the mandal level is one such aggregation.
- **HaritGlo** (agricultural MRV module): Stress trajectories feed yield forecasts and crop-condition signals for carbon-credit MRV in regenerative agriculture programmes.
- **Ātmanetra** (orbital edge AI module): The spectral-downsampling study planned as a follow-up will inform the band-selection design for hyperspectral payloads on small-satellite missions.

## Acknowledgements

Hyperspectral data provided by **IIT Hyderabad TiHAN** (UC-HSI Crop Variety Dataset — Groundnut Water Stress component). All technical analysis, modelling, and platform integration by Aganitha Space Technologies Pvt. Ltd. Analysis performed on Google Colab. The authors thank IIT-H TiHAN for the dataset and for ongoing dialogue on extension to additional crops.

## Contact

Narasimha Sharma Narayanam · Founder, Chairman & Managing Director
Aganitha Space Technologies Pvt. Ltd.
nsnarayanam@aganithaspace.com
