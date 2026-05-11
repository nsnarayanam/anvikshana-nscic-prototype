<img width="1196" height="820" alt="AgST logo" src="https://github.com/user-attachments/assets/9562afac-42f1-47a8-a07d-f4b881811b74" />
# Anvīkṣaṇa — Agricultural Drought Intelligence
### NSCIC Stage 2 Prototype | National Climate Stack Innovation Challenge

**Aganitha Space Technologies Pvt. Ltd.** · Secunderabad, Hyderabad · May 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: DiCRA](https://img.shields.io/badge/Data-DiCRA%20%2F%20UNDP-blue.svg)](https://dicra.undp.org.in)
[![OGC CSAPI](https://img.shields.io/badge/Standard-OGC%20CSAPI-orange.svg)](https://ogcapi.ogc.org)

---

## What This Is

Anvīkṣaṇa is an agricultural drought intelligence platform built on real satellite data from [DiCRA (UNDP India)](https://dicra.undp.org.in) — a Digital Public Good. It ingests mandal-level NDVI, soil moisture, and land surface temperature data, computes internationally recognised drought indices, and projects district-level drought probability 15 years forward under SSP2-4.5 and SSP5-8.5 climate scenarios.

Built for NABARD and climate-resilient lending — giving lenders a science-based drought risk signal before loan origination, not after crop loss.

---

## Live Dashboard

🚀 **[Launch Anvīkṣaṇa Dashboard →](https://share.streamlit.io)**

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Real DiCRA records processed | **487,243** |
| Mandal polygons (real boundaries) | **592** |
| Districts covered | **33 (Telangana)** |
| Biweekly NDVI observations | **23 dates · 2025** |
| Model ROC-AUC (spatial CV) | **0.974 ± 0.004** |
| F1 Score | **0.801 ± 0.032** |
| Brier Score | **0.058** |
| Projection horizon | **2026–2040** |
| Climate scenarios | **SSP2-4.5 · SSP5-8.5** |

---

## Architecture — 4 Layers

```
Layer 1 — Data Inputs
    DiCRA NDVI vectors (592 mandal polygons, 23 biweekly dates)
    DiCRA Soil Moisture Index (576 mandals, 12 months)
    DiCRA Land Surface Temperature (H3 Res-7 indexed)
    H3 DGGS indexing at Resolution 7 (21,363 cells)

Layer 2 — Modelling Engine
    VCI (Vegetation Condition Index) — FAO / ISRO standard
    CDSI (Combined Drought Severity Index) = 0.6×VCI + 0.4×SMDI
    Random Forest Classifier (binary drought detection)
    Gradient Boosting Regressor (continuous CDSI)
    16 engineered features: lag-1/3/6, rolling-3/6, NDVI×SM interaction
    Spatial cross-validation: GroupKFold (k=5), district hold-out

Layer 3 — Forward Projections (2026–2040)
    IPCC AR6 WG1 South Asia regional change factors
    SSP2-4.5 and SSP5-8.5 scenarios
    20-member ensemble for uncertainty quantification
    District-level drought probability surfaces

Layer 4 — Application Layer
    Interactive Streamlit dashboard (5 pages)
    Real mandal-level choropleth maps (DiCRA polygon boundaries)
    Seasonal time series, scenario explorer, model validation
    OGC CSAPI compliant · Standards-native architecture
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Overview | KPI cards · district scatter map · seasonal NDVI cycle · feature importance |
| Mandal Drought Map | Real polygon choropleth · 592 DiCRA boundaries · date slider · NDVI/SM/CDSI toggle |
| Seasonal Analysis | District time series · all-district NDVI heatmap · correlation analysis |
| 2026–2040 Projections | Year slider · scenario toggle · uncertainty bands · SSP2 vs SSP5 |
| Model Validation | ROC-AUC · Brier Score · spatial CV methodology · data provenance |

---

## Drought Indices

### Vegetation Condition Index (VCI)
```
VCI = (NDVI − NDVIₘᵢₙ) / (NDVIₘₐₓ − NDVIₘᵢₙ) × 100
```
- Computed per mandal using 2025 historical NDVI envelope (23 biweekly dates)
- 0 = Extreme Drought · 100 = No Drought
- Standard used by FAO, ISRO FASAL programme, NOAA Drought Monitor

| VCI Range | Class |
|-----------|-------|
| 0–10 | Extreme Drought |
| 10–20 | Severe Drought |
| 20–35 | Moderate Drought |
| 35–50 | Watch |
| 50+ | No Drought |

### Key Finding — 2025 Telangana
- **April VCI = 17.9** → Severe Drought (pre-monsoon stress)
- **September VCI = 89.6** → No Drought (Kharif peak)
- **7.4× seasonal swing** confirms DiCRA data quality
- **Most vulnerable districts**: Medak, Mahabubnagar, Narayanpet, Nagarkurnool

---

## Data Sources

All data sourced from **DiCRA (Data in Climate Resilient Agriculture)** — an open Digital Public Good facilitated by UNDP India and Government of Telangana.

| Dataset | Source | Records |
|---------|--------|---------|
| NDVI Vectors (mandal polygons) | DiCRA / UNDP India | 13,616 (592×23) |
| NDVI H3 Grid (Res-7) | DiCRA / UNDP India | 427,260 |
| Soil Moisture | DiCRA / UNDP India | 56,365 |
| Land Surface Temperature | DiCRA / UNDP India | H3 indexed |
| **Total** | **DiCRA (UNDP India)** | **487,243** |

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/anvikshana-nscic-prototype.git
cd anvikshana-nscic-prototype
pip install -r requirements.txt
streamlit run dashboard.py
```

Open `http://localhost:8501`

---

## Deploy to Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → Main file: `dashboard.py`
4. Deploy → live URL in ~3 minutes

---

## Standards & Compliance

- **OGC CSAPI** — Climate Stack API standards (active contributor, OGC SWG)
- **H3 DGGS** — Uber H3 Discrete Global Grid System at Resolution 7
- **IEEE GRSS** — Earth Observation standards alignment
- **W3C WoT** — Web of Things working group
- **DiCRA** — UN Digital Public Good principles (open data, open code, open API)

---

## About Anvīkṣaṇa

Anvīkṣaṇa is the drought intelligence module of the Aganitha Space Technologies Earth Intelligence Platform. The platform consists of six modules:

- **Arunam** — Solar & air quality intelligence (Solar Impulse Global Efficient Solution label)
- **HaritGlo** — Carbon MRV
- **Pavanapatha** — Atmospheric & multi-hazard monitoring
- **Samudranetra** — Ocean & coastal intelligence
- **INDRANET** — IoT sensor mesh
- **Ātmanetra** — Orbital edge AI

---

---

## Complementary capability module — Hyperspectral Water Stress Detection

The [`modules/hyperspectral_water_stress/`](modules/hyperspectral_water_stress/) folder contains a pixel-level hyperspectral water-stress classifier (groundnut, 99.96% test accuracy, validated on the IIT-H TiHAN UC-HSI dataset) that demonstrates Anvīkṣaṇa's multi-scale architecture — from individual canopy spectra to mandal-level aggregation. The hyperspectral classifier complements the main mandal-level NSCIC dashboard at a different sensor modality and spatial scale, feeding finer-grained ground-truth signals into the same DGGS-indexed knowledge graph. See the [module README](modules/hyperspectral_water_stress/README.md) for full results, methodology, and data licensing.


## About Aganitha Space Technologies

Aganitha Space Technologies Pvt. Ltd. is a deep-tech company based in Secunderabad, Hyderabad, building standards-native Earth Intelligence infrastructure for climate resilience, food security, and sustainable development.

- DPIIT Recognised Startup (DIPP162965)
- IIT Bombay Research Partnership 
- Active in OGC, IEEE GRSS, W3C WoT standards bodies
- Solar Impulse Foundation Global Efficent Member

**Contact:** nsnarayanam@aganithaspace.com  
**Website:** [www.aganithaspace.com](https://www.aganithaspace.com)

---

## Citation

If you use this work, please cite:

```
Narayanam, N.S. (2026). Anvīkṣaṇa: Agricultural Drought Intelligence 
for Climate-Resilient Lending. NSCIC Stage 2 Prototype. 
Aganitha Space Technologies Pvt. Ltd., Hyderabad, India.
Data: DiCRA / UNDP India (Digital Public Good).
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.  
Data sourced from DiCRA is subject to [DiCRA Terms of Service](https://dicra.undp.org.in).
