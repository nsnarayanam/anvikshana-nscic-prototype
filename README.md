<p align="center">
  <img src="AgST logo.png" alt="Aganitha Space Technologies" width="180"/>
</p>

<h1 align="center">Anvīkṣaṇa — Agricultural Drought Intelligence</h1>

<p align="center">
  <strong>NSCIC Stage 3 Finalist | National Climate Stack Innovation Challenge</strong><br/>
  <strong>Aganitha Space Technologies Pvt. Ltd.</strong> · Secunderabad, Hyderabad · June 2026
</p>

<p align="center">
  <a href="https://anvikshana-nscic-prototype-863x8w2jstcmawtjtrvrvk.streamlit.app">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App"/>
  </a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/></a>
  &nbsp;
  <a href="https://dicra.undp.org.in"><img src="https://img.shields.io/badge/Data-DiCRA%20%2F%20UNDP-blue.svg" alt="DiCRA"/></a>
  &nbsp;
  <img src="https://img.shields.io/badge/Climate-CORDEX--SA-teal.svg" alt="CORDEX-SA"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Standard-OGC%20CSAPI-orange.svg" alt="OGC CSAPI"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Standard-IEEE%20P4011-lightblue.svg" alt="IEEE P4011"/>
</p>

---

## 🌐 Live Dashboard

**[→ Launch Anvīkṣaṇa Dashboard](https://anvikshana-nscic-prototype-863x8w2jstcmawtjtrvrvk.streamlit.app)**

---

## What This Is

Anvīkṣaṇa is a mandal-level agricultural drought intelligence platform built on real satellite data from [DiCRA / UNDP India](https://dicra.undp.org.in) — a Digital Public Good. It ingests NDVI, soil moisture, and land surface temperature data across 592 Telangana mandals, computes internationally recognised drought indices (VCI, CDSI), and projects district-level drought probability 15 years forward under RCP/SSP climate scenarios, driven by **CORDEX-SA downscaled regional climate model data** (IITM-RegCM4-4, Copernicus CDS).

**Primary use case:** Pre-loan drought risk scoring for NABARD KCC disbursement — delivering a 45–60 day early warning over existing approaches, at 5× finer spatial resolution than district-level SPI.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Real DiCRA records processed | **487,243** |
| Mandal polygons (real DiCRA boundaries) | **592** |
| Districts covered | **33 (Telangana)** |
| Biweekly NDVI observation dates | **23 dates · 2025 + 23 dates · 2018** |
| Model ROC-AUC (spatial cross-validation) | **0.974 ± 0.004** |
| F1 Score | **0.801 ± 0.032** |
| Brier Score | **0.058** |
| 2018 drought year detection | **100% — all 33 declared districts** |
| Benchmark vs SPI-3 baseline | **+25pp accuracy · +45 days lead time** |
| Projection horizon | **2026–2040** |
| Climate projection source | **CORDEX-SA · IITM-RegCM4-4 · Copernicus CDS** |
| Climate scenarios | **RCP 4.5 · RCP 8.5 / SSP2-4.5 · SSP5-8.5** |
| Ensemble members | **20** |
| Scalability | **Andhra Pradesh proven · 4 states ready** |

---

## Architecture — 4 Layers

​```
Layer 1 — Data Inputs (100% Real)
    DiCRA NDVI vectors (592 mandal polygons · 23 biweekly dates · 2025 + 2018)
    DiCRA Soil Moisture Index (576 mandals · 12 months)
    DiCRA Land Surface Temperature (H3 Res-7 indexed · 21,363 cells)
    NASA POWER GMAO 2000–2024 (25-year climate baseline)
    IMD Live Rainfall API (real-time validation)

Layer 2 — Modelling Engine
    VCI = (NDVI − NDVImin) / (NDVImax − NDVImin) × 100  [FAO/ISRO standard]
    CDSI = 0.6 × VCI + 0.4 × SMDI
    Random Forest Classifier (binary drought detection, 200 trees)
    Gradient Boosting Regressor (continuous CDSI, 200 estimators)
    16 engineered features: lag-1/3/6, rolling-3/6, NDVI×SM interaction
    Spatial cross-validation: GroupKFold (k=5), district hold-out

Layer 3 — Forward Projections (2026–2040)
    CORDEX-SA downscaled RCM data (IITM-RegCM4-4, Copernicus CDS, 0.44° / ~50km)
    RCP 4.5 (moderate) and RCP 8.5 (high emissions) scenarios
    91 grid cells over Telangana · daily data aggregated to monthly/annual
    20-member ensemble uncertainty quantification
    District-level drought probability surfaces

Layer 4 — Application & API Layer
    11-page interactive Streamlit dashboard (live URL above)
    OGC CSAPI-compliant API endpoint (ogc_api.py)
    DiCRA 2.0 contribution layer (GeoJSON FeatureCollection)
    NABARD pre-loan risk scoring interface
​```

---

## Dashboard Pages (11)

| Page | Description |
|------|-------------|
| 📊 Overview | KPI cards · district scatter map · seasonal NDVI cycle · feature importance |
| 🗺️ Mandal Drought Map | Real polygon choropleth · 592 DiCRA boundaries · date slider · NDVI/SM/CDSI toggle |
| 📈 Seasonal Analysis | District time series · all-district NDVI heatmap · correlation r=0.45 |
| 🔁 2018 vs 2025 Comparison | Historical drought validation · 33 districts · monthly NDVI delta |
| 🌧️ NASA POWER & SPI | 25-year climate baseline · SPI-3/6 · drought climatology 2000–2024 |
| 🚨 IMD Live Alerts | Real-time rainfall deficit · live validation against model predictions |
| 🔮 2026–2040 Projections | Year slider · SSP scenario toggle · uncertainty bands · top at-risk districts |
| 🌡️ CORDEX-SA Projections | Real downscaled RCM data · IITM-RegCM4-4 · temperature & rainfall trajectories |
| 🤖 Model Validation | ROC-AUC · Brier · spatial CV methodology · data provenance |
| 🌾 Crop Yield Risk Score | One number per mandal · traffic-light NABARD pre-loan decision tool |
| 🔌 OGC API Explorer | Live GeoJSON responses · EDR position query · benchmark vs SPI-3 · AP scalability |

---

## Climate Projections — CORDEX-SA

Drought projections (2026–2040) are driven by **CORDEX-SA downscaled regional climate model data** from the IITM-RegCM4-4 model (Indian Institute of Tropical Meteorology, Pune), sourced from the Copernicus Climate Data Store at 0.44° (~50km) resolution. Both RCP 4.5 and RCP 8.5 scenarios are integrated, covering 91 grid cells over Telangana, with daily data aggregated to monthly and annual time steps. This replaces IPCC scaling factors with physically-simulated regional climate output.

**Data source:** CORDEX-SA · IITM-RegCM4-4 · MPI-M-MPI-ESM-MR · Copernicus CDS · DOI 10.24381/cds.bc91edc3

---

## OGC API Endpoint

`ogc_api.py` implements a fully OGC-compliant REST API:

​```bash
# Run locally
pip install fastapi uvicorn
uvicorn ogc_api:app --host 0.0.0.0 --port 8502 --reload
​```

| Endpoint | Description |
|----------|-------------|
| `GET /` | Landing page (OGC conformance) |
| `GET /conformance` | 7 OGC conformance classes declared |
| `GET /collections` | drought-index · drought-projections |
| `GET /collections/drought-index/items` | Current mandal VCI/CDSI as GeoJSON FeatureCollection |
| `GET /collections/drought-projections/items` | 2026–2040 district projections · SSP2/SSP5 |
| `GET /position?coords=POINT(lon lat)` | OGC API-EDR nearest-mandal drought value |
| `GET /health` | Data provenance · model metrics · standards conformance |

**Conforms to:** OGC API-Features 1.0 · OGC API-EDR 1.1 · OGC CSAPI draft · W3C WoT/SSN · ISO 19179 (under review)

---

## Historical Validation — 2018 vs 2025

| Metric | 2018 (Declared Drought) | 2025 (Current) |
|--------|------------------------|----------------|
| Mean VCI (cross-year) | 36.9 — Watch/Moderate | 52.6 — No Drought |
| Driest month VCI | May: 12.8 (Severe Drought) | April: 28.8 |
| Model detection | ✅ All 33 districts flagged | — |
| False negatives | 0 | — |

**2018 is a government-declared drought year (all 33 Telangana districts).** The model was trained on 2025 data and tested blind on 2018 labels — 100% detection rate.

---

## Benchmark vs SPI-3 (WMO Standard)

| Metric | SPI-3 Baseline | Anvīkṣaṇa |
|--------|---------------|-----------|
| Overall accuracy (8 years) | 75% | **100%** |
| Drought detection rate | 80% | **100%** |
| False alarm rate | 67% | **0%** |
| Lead time before crop loss | 0 days | **45–60 days** |
| Spatial resolution | District | **Mandal (592)** |
| Forward projection | None | **2026–2040** |
| ROC-AUC | N/A | **0.974 ± 0.004** |

---

## Scalability — Andhra Pradesh Proven

The same pipeline runs on Andhra Pradesh with zero code changes:
- 13 AP districts · 4 agro-climatic zones · 390 projection rows
- Rayalaseema: 59.7% drought probability by 2040 (SSP5-8.5)
- Next states: Maharashtra · Karnataka · Madhya Pradesh · Odisha

---

## Repository Structure

​anvikshana-nscic-prototype/

├── dashboard.py                        # 11-page Streamlit dashboard

├── ogc_api.py                          # OGC CSAPI-compliant FastAPI endpoint

├── requirements.txt                    # Python dependencies

├── mandal_ndvi_2018_full.csv           # 2018 historical NDVI

├── district_comparison_2018_2025.csv   # 2018 vs 2025 comparison

├── cordex_telangana_monthly.csv        # CORDEX-SA climate data

├── cordex_projections_page.py          # CORDEX-SA dashboard module

├── AgST logo.png                       # Aganitha logo

├── data/geojson_ndvi/                  # 2025 GeoJSON + climate CSVs

├── outputs/                            # projections, metrics, benchmarks

└── scripts/                            # data pipeline + model training

---

## Standards Body Participation

Aganitha Space Technologies is an active contributor to:

- **OGC CSAPI Standards Working Group** — Anvīkṣaṇa's UAS Analytics page inspired the collaborative CSAPI Explorer at [ogc-csapi-explorer.pages.dev](https://ogc-csapi-explorer.pages.dev/analytics)
- **IEEE GRSS P4011** — Voting Member (Disaster Intelligence and Earth Observation standards)
- **W3C Web of Things Working Group**
- **ISO/TC 211** — Formal comment submitted on ISO/CD TR 19179

---

## IP & Legal

- Core methodology: 4 Indian patents filed · PCT filing planned
- Trademarks filed
- This repository is licensed under MIT. Use requires attribution to Aganitha Space Technologies Pvt. Ltd.
- *Patent pending. Platform name Anvīkṣaṇa is a trademark of Aganitha Space Technologies Pvt. Ltd.*

---

## Organisation

**Aganitha Space Technologies Pvt. Ltd.**
Secunderabad, Hyderabad, Telangana, India

**Founder & MD:** Narasimha Sharma Narayanam
OGC CSAPI SWG · IEEE GRSS P4011 · W3C WoT WG · Royal Aeronautical Society (Associate) · IET (Fellow)

📧 nsnarayanam@aganithaspace.com
🌐 [Live Dashboard](https://anvikshana-nscic-prototype-863x8w2jstcmawtjtrvrvk.streamlit.app)

---

*Data: DiCRA / UNDP India (Digital Public Good) · NASA POWER / GMAO · India Meteorological Department · CORDEX-SA / Copernicus CDS*
*Model: Aganitha Space Technologies Pvt. Ltd. · NSCIC Stage 3 Finalist · June 2026*
