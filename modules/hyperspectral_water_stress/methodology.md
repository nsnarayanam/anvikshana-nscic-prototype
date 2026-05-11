# Hyperspectral Water Stress Detection in Groundnut

## Methodology and Results — Aganitha Space Technologies / Anvīkṣaṇa Platform

**Author:** Narasimha Sharma Narayanam
**Affiliation:** Aganitha Space Technologies Pvt. Ltd.
**Dataset source:** IIT Hyderabad TiHAN — UC-HSI Crop Variety Dataset (Groundnut Water Stress component)
**Version:** May 2026
**Status:** Preliminary, internally-validated. NSCIC Stage 2 supporting evidence.

---

## 1. Problem statement

Early detection of crop water stress is a foundational component of climate-resilient agriculture. Conventional broadband vegetation indices (NDVI, EVI) derived from multispectral satellites detect stress only after canopy structure has already degraded — typically days or weeks after the underlying water deficit begins. Hyperspectral imaging (HSI), with its continuous narrow-band sampling across the 400–2500 nm range, can in principle resolve the subtle biophysical changes that precede visible symptoms, but the signal of interest is distributed across many spectral bands and is not separable by simple threshold rules. This study evaluates whether multivariate machine-learning models applied to pixel-level hyperspectral signatures can recover that signal with the accuracy and physical interpretability needed for operational deployment as a foundational layer in the Anvīkṣaṇa Earth Intelligence platform.

## 2. Data

Pixel-level hyperspectral reflectance measurements over groundnut canopies were obtained from the IIT Hyderabad TiHAN UC-HSI Crop Variety Dataset (Groundnut Water Stress component). The dataset comprises 16,667 reflectance vectors of 300 spectral bands each, drawn from two acquisition conditions corresponding to a binary stress label: well-watered (class 0, n = 7,954) and water-stressed (class 1, n = 8,713). Reflectance values are physically plausible (range 0.016–0.677, mean 0.208) with no missing or out-of-range entries. Class assignment to well-watered vs. stressed is inferred from the mean spectral signatures, which show the canonical NIR-plateau depression in class 1 consistent with reduced leaf mesophyll structure under water deficit.

Data were split into stratified train/validation/test sets (70/15/15) with a fixed random seed (42). Train-set statistics were used for per-band standardisation, applied identically to validation and test partitions to prevent information leakage.

## 3. Methods

Three classifier families were benchmarked on the same standardised splits to characterise the structure of the discriminative signal:

1. **Random Forest** (300 trees, balanced class weights) — tree-based ensemble, providing per-band feature-importance rankings.
2. **Support Vector Machine** with radial-basis-function kernel (C = 10, balanced class weights) — the established classical baseline for hyperspectral classification.
3. **One-dimensional convolutional neural network** — three convolutional blocks (32→64→128 channels, kernel sizes 7-5-3, batch normalisation, ReLU activation, max-pooling), followed by adaptive average pooling and a two-layer fully-connected classifier with dropout (0.3). Trained for 40 epochs with the AdamW optimiser (learning rate 1e-3, weight decay 1e-4) and cosine-annealing learning-rate schedule.

To probe the dimensionality of the signal, four additional baseline classifiers were evaluated:

- Logistic regression on individual top-ranked bands (161, 180, 160, 2, 259)
- Logistic regression on overall canopy brightness (mean reflectance across all bands)
- Logistic regression on the full 300-band spectrum (linear multivariate)

Per-band two-sample t-tests of class-conditional reflectance distributions identified the regions of strongest discriminative signal.

## 4. Results

### 4.1 Classification performance

| Model | Test accuracy | F1 (macro) | ROC-AUC | Errors / 2,501 |
|---|---|---|---|---|
| Random Forest | 0.9744 | 0.9743 | 0.9976 | 64 |
| SVM (RBF) | 0.9988 | 0.9988 | 1.0000 | 3 |
| 1D-CNN | 0.9996 | 0.9996 | 0.9999 | 1 |
| Logistic regression (full spectrum) | 0.9992 | — | — | ~2 |
| Logistic regression (band 161 only) | 0.7273 | — | — | — |
| Logistic regression (brightness only) | 0.6253 | — | — | — |

### 4.2 Structure of the discriminative signal

Mean spectral signatures of the two classes overlap within one standard deviation across the full 400–2500 nm range, with grand-mean reflectance differing by only 3.6% (0.212 vs. 0.205). Univariate classifiers achieve only 57–73% accuracy, and overall canopy brightness is essentially non-discriminative (62.5%). Yet multivariate models that integrate information across all 300 bands — linear (logistic regression, 99.92%), kernel-based (SVM-RBF, 99.88%), and deep (1D-CNN, 99.96%) — all converge at near-perfect separation. **The signal is therefore distributed, multivariate, and recoverable by both linear and non-linear integration of the full spectrum.**

Per-band t-statistics localise the strongest discriminative signal to three spectrally meaningful regions:

- **Bands 158–180 (red edge / NIR transition):** dominant region of discrimination. Band 161 yields the highest t-statistic (|t| = 76.0); band 180 shows the largest absolute reflectance difference (0.049). This region corresponds to the canonical "red-edge position" (REP), which shifts toward shorter wavelengths under water stress — a well-documented biophysical mechanism driven by chlorophyll degradation and changes in leaf internal scattering.
- **Bands 75–90 (visible green/yellow edge):** secondary discriminative region (|t| ≈ 30–50), consistent with early chlorophyll-pigment changes.
- **Bands 200–300 (NIR plateau and SWIR transition):** sustained low-amplitude difference (|t| ≈ 20–30), reflecting the overall NIR-plateau depression in stressed leaves.

The convergence of all three classifier families, combined with the physical interpretability of the dominant discriminative bands, and the **independent agreement between t-statistic ranking and Random Forest feature importance**, indicates that the result is not a model-overfitting artefact but a genuine biophysical signature recovered by spectroscopy.

## 5. Strategic positioning within Anvīkṣaṇa

This per-pixel hyperspectral classifier is the **lowest-level capability layer** of Anvīkṣaṇa, Aganitha's Earth Intelligence Operating System. The classifier integrates with the broader stack as follows:

- **Arunam** (climate-risk module): Per-pixel water-stress detections aggregate into DGGS-indexed regional stress indices, fused with satellite-derived soil-moisture and rainfall anomalies to generate operational crop-stress alerts at the panchayat / mandal level — the spatial scale used in the main NSCIC dashboard (`dashboard.py` in this repository).
- **HaritGlo** (agricultural MRV module): Stress detections inform crop-condition trajectories that feed into yield forecasts and the carbon-credit MRV pipeline for regenerative agriculture programmes.
- **Ātmanetra** (orbital edge AI module): The spectral-downsampling study planned as a follow-up will inform band-selection design for hyperspectral payloads on small-satellite missions.
- **Cross-platform transferability:** The same 1D-CNN architecture is being extended to other crops in the UC-HSI dataset family (pearl millet, additional groundnut acquisitions), with the goal of producing a transferable hyperspectral water-stress model for India's principal rainfed crops.

## 6. Caveats and next steps

**Caveats applicable to the current result:**

1. **Pixel-level random split.** Train and test pixels may originate from spatially adjacent locations within the same plant or canopy region. The current result therefore characterises the *spectral separability* of stressed vs. well-watered conditions rather than the model's ability to generalise to unseen plants or unseen acquisition dates.
2. **Single-site, single-cultivar dataset.** Generalisation across cultivars, soil types, growth stages, and illumination conditions has not yet been evaluated.
3. **Binary stress label.** The dataset encodes stress as a binary state; operational deployment will require a continuous stress severity index or multi-class severity grades.

**Planned validation extensions:**

1. **Leave-one-plant-out cross-validation** once plant-ID metadata is obtained from the dataset originator (IIT-H TiHAN).
2. **Held-out acquisition-date validation** to assess robustness to temporal and illumination variability.
3. **Transfer evaluation** to additional UC-HSI crop datasets (pearl millet, sorghum), with assessment of whether fine-tuning is necessary or whether the red-edge / NIR-plateau signature transfers zero-shot across crop types.
4. **Spectral downsampling study** to determine the minimum band set sufficient for accurate stress detection — directly relevant for designing operational hyperspectral payloads under the Ātmanetra orbital edge AI module.

## 7. Reproducibility

All code, trained model weights, and result tables are stored or referenced from this repository:

- `pipeline.py` — full reproducible pipeline
- `results.json` — full numerical results
- `figures/` — diagnostic figures (mean spectra, difference spectrum, RF importance, summary)
- Trained CNN weights (`groundnut_water_stress_cnn.pt`) held privately by Aganitha pending TiHAN redistribution permission — available on request (see `DATA_LICENSING.md`)

Random seed: 42. Software stack: Python 3.12, NumPy, scikit-learn, PyTorch.

---

**Suggested citation.** Narayanam, N. S. (2026). *Hyperspectral water-stress detection in groundnut: a distributed-signal multivariate classification study.* Aganitha Space Technologies / Anvīkṣaṇa platform technical report, NSCIC Stage 2 supporting evidence.

**Acknowledgement.** Hyperspectral data provided by IIT Hyderabad TiHAN (UC-HSI Crop Variety Dataset). Analysis performed on Google Colab. The authors thank IIT-H TiHAN for the dataset and for ongoing dialogue on extension to additional crops.
