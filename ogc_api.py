"""
Anvīkṣaṇa — OGC Climate Stack API (CSAPI) Endpoint
====================================================
OGC CSAPI-compliant REST API exposing Anvīkṣaṇa drought intelligence
outputs back to DiCRA 2.0 and the National Climate Stack ecosystem.

Conforms to:
  - OGC API - Environmental Data Retrieval (EDR) v1.1
  - OGC CSAPI (Climate Stack API) draft specification
  - OGC API - Features (GeoJSON output)
  - W3C WoT (sensor metadata)

Usage:
    uvicorn ogc_api:app --host 0.0.0.0 --port 8502 --reload

Endpoints:
    GET /                          → Landing page (OGC conformance)
    GET /conformance               → OGC conformance classes
    GET /collections               → Available data collections
    GET /collections/drought-index → Collection metadata
    GET /collections/drought-index/items          → Current mandal VCI/CDSI (GeoJSON)
    GET /collections/drought-index/items/{uid}    → Single mandal
    GET /collections/drought-projections/items    → 2026-2040 district projections
    GET /collections/drought-projections/items/{district}/{scenario}/{year}
    GET /position?coords=POINT(lon lat)&parameter-name=cdsi  → EDR position query
    GET /health                    → Health check + data provenance

Aganitha Space Technologies Pvt. Ltd. · nsnarayanam@aganithaspace.com
OGC CSAPI SWG active contributor · IEEE GRSS P4011 Voting Member
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent
DATA_DIR  = BASE / "data" / "geojson_ndvi"
OUT_DIR   = BASE / "outputs"

# ── Load data once at startup ─────────────────────────────────────────────────
merged = pd.read_csv(DATA_DIR / "mandal_ndvi_sm_merged.csv")
proj   = pd.read_csv(OUT_DIR  / "drought_projections_2026_2040.csv")
val    = pd.read_csv(OUT_DIR  / "historical_drought_validation.csv")

# Latest date slice for current conditions
merged["date"] = pd.to_datetime(merged["date"])
latest_date = merged["date"].max()
latest      = merged[merged["date"] == latest_date].copy()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Anvīkṣaṇa OGC CSAPI",
    description="OGC Climate Stack API — Agricultural Drought Intelligence for DiCRA 2.0",
    version="1.0.0",
    contact={"name": "Aganitha Space Technologies", "email": "nsnarayanam@aganithaspace.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

NOW = lambda: datetime.now(timezone.utc).isoformat()

# ── Helper: build GeoJSON Feature from a merged row ───────────────────────────
def mandal_to_feature(row):
    return {
        "type": "Feature",
        "id": row["uid"],
        "geometry": {
            "type": "Point",
            "coordinates": [round(row["longitude"], 5), round(row["latitude"], 5)]
        },
        "properties": {
            "uid":           row["uid"],
            "mandal":        row["mandal"],
            "district":      row["district"],
            "date":          str(row["date"])[:10],
            "ndvi_mean":     round(row["ndvi_mean"], 4),
            "sm_mean":       round(row["sm_mean"],   4),
            "cdsi":          round(row["cdsi"],      4),
            "drought_class": row["drought_class"],
            "area_km2":      round(row["area_km2"],  2),
            # OGC EDR standard fields
            "parameter_names": ["ndvi_mean", "sm_mean", "cdsi"],
            "phenomenon_time": str(row["date"])[:10],
            "result_quality":  "DiCRA/UNDP validated satellite data",
        }
    }

# ── OGC Landing Page ──────────────────────────────────────────────────────────
@app.get("/", tags=["OGC"])
def landing_page():
    return {
        "title":       "Anvīkṣaṇa Agricultural Drought Intelligence API",
        "description": "OGC CSAPI-compliant endpoint exposing mandal-level drought indices "
                       "and 15-year projections for DiCRA 2.0 integration. "
                       "Aganitha Space Technologies Pvt. Ltd.",
        "attribution": "Data: DiCRA / UNDP India (Digital Public Good) · "
                       "Model: Aganitha Space Technologies",
        "links": [
            {"href": "/conformance",                            "rel": "conformance",   "type": "application/json", "title": "OGC conformance classes"},
            {"href": "/collections",                            "rel": "data",          "type": "application/json", "title": "Data collections"},
            {"href": "/collections/drought-index/items",        "rel": "items",         "type": "application/geo+json", "title": "Current drought index (all mandals)"},
            {"href": "/collections/drought-projections/items",  "rel": "items",         "type": "application/geo+json", "title": "2026-2040 projections"},
            {"href": "/health",                                 "rel": "status",        "type": "application/json", "title": "Health & data provenance"},
        ]
    }

# ── OGC Conformance ───────────────────────────────────────────────────────────
@app.get("/conformance", tags=["OGC"])
def conformance():
    return {
        "conformsTo": [
            "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/core",
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson",
            "http://www.opengis.net/spec/ogcapi-edr-1/1.1/conf/core",
            "http://www.opengis.net/spec/ogcapi-edr-1/1.1/conf/geojson",
            "https://ogcapi.ogc.org/csapi/conf/core",           # OGC CSAPI
            "https://www.w3.org/TR/vocab-ssn/",                 # W3C SSN/SOSA
        ]
    }

# ── Collections ───────────────────────────────────────────────────────────────
@app.get("/collections", tags=["OGC"])
def collections():
    return {
        "collections": [
            {
                "id":          "drought-index",
                "title":       "Anvīkṣaṇa Drought Index — Current Conditions",
                "description": "Mandal-level VCI (Vegetation Condition Index) and CDSI "
                               "(Combined Drought Severity Index) derived from DiCRA NDVI "
                               "and soil moisture. 592 mandals, Telangana, 2025.",
                "extent": {
                    "spatial":  {"bbox": [[76.8, 15.8, 81.4, 19.9]], "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "temporal": {"interval": [["2025-01-01", str(latest_date)[:10]]], "trs": "http://www.opengis.net/def/uom/ISO-8601/0/Gregorian"}
                },
                "itemType": "feature",
                "crs":      ["http://www.opengis.net/def/crs/OGC/1.3/CRS84"],
                "links": [{"href": "/collections/drought-index/items", "rel": "items", "type": "application/geo+json"}],
                "parameters": {
                    "ndvi_mean":     {"description": "Mean NDVI from DiCRA satellite data",   "unit": "dimensionless [0-1]",   "source": "DiCRA/UNDP India"},
                    "sm_mean":       {"description": "Mean soil moisture index",              "unit": "dimensionless [0-1]",   "source": "DiCRA/UNDP India"},
                    "cdsi":          {"description": "Combined Drought Severity Index (0.6×VCI + 0.4×SMDI)", "unit": "dimensionless", "source": "Aganitha model"},
                    "drought_class": {"description": "Drought classification",               "values": ["No Drought","Watch","Moderate","Severe","Extreme"]},
                }
            },
            {
                "id":          "drought-projections",
                "title":       "Anvīkṣaṇa 15-Year Drought Projections 2026–2040",
                "description": "District-level drought probability projections under SSP2-4.5 "
                               "and SSP5-8.5. Random Forest model + IPCC AR6 WG1 South Asia "
                               "regional change factors. 20-member ensemble uncertainty.",
                "extent": {
                    "spatial":  {"bbox": [[76.8, 15.8, 81.4, 19.9]], "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "temporal": {"interval": [["2026-01-01", "2040-12-31"]], "trs": "http://www.opengis.net/def/uom/ISO-8601/0/Gregorian"}
                },
                "itemType": "feature",
                "crs":      ["http://www.opengis.net/def/crs/OGC/1.3/CRS84"],
                "links": [{"href": "/collections/drought-projections/items", "rel": "items", "type": "application/geo+json"}],
                "parameters": {
                    "drought_probability": {"description": "Probability of drought conditions", "unit": "[0-1]"},
                    "scenario":            {"description": "IPCC SSP scenario",                "values": ["SSP2-4.5", "SSP5-8.5"]},
                    "proj_temp_c":         {"description": "Projected mean temperature",       "unit": "°C"},
                    "proj_spi3":           {"description": "Projected SPI-3 index",            "unit": "standardised"},
                }
            }
        ]
    }

# ── Drought Index Items (all mandals, GeoJSON FeatureCollection) ──────────────
@app.get("/collections/drought-index/items", tags=["Drought Index"])
def drought_index_items(
    district: str = Query(None, description="Filter by district name"),
    drought_class: str = Query(None, description="Filter: No Drought | Watch | Moderate | Severe | Extreme"),
    limit: int = Query(100, ge=1, le=600, description="Max features to return"),
    offset: int = Query(0, ge=0),
):
    df = latest.copy()
    if district:
        df = df[df["district"].str.lower() == district.lower()]
        if df.empty:
            raise HTTPException(404, f"District '{district}' not found")
    if drought_class:
        df = df[df["drought_class"].str.lower() == drought_class.lower()]

    total = len(df)
    df = df.iloc[offset: offset + limit]

    features = [mandal_to_feature(row) for _, row in df.iterrows()]

    return {
        "type":            "FeatureCollection",
        "timeStamp":       NOW(),
        "numberMatched":   total,
        "numberReturned":  len(features),
        "features":        features,
        "links": [
            {"href": "/collections/drought-index/items", "rel": "self",  "type": "application/geo+json"},
            {"href": "/collections/drought-index",       "rel": "collection", "type": "application/json"},
        ],
        "_anvikshana_meta": {
            "observation_date": str(latest_date)[:10],
            "total_mandals":    int(latest["mandal"].nunique()),
            "total_districts":  int(latest["district"].nunique()),
            "data_source":      "DiCRA / UNDP India — Digital Public Good",
            "model_version":    "Anvīkṣaṇa v1.0 — NSCIC Stage 2",
            "standards":        ["OGC CSAPI", "OGC API-Features", "OGC API-EDR", "W3C WoT"],
        }
    }

# ── Single mandal by UID ──────────────────────────────────────────────────────
@app.get("/collections/drought-index/items/{uid}", tags=["Drought Index"])
def drought_index_item(uid: str):
    row = latest[latest["uid"] == uid]
    if row.empty:
        raise HTTPException(404, f"Mandal UID '{uid}' not found")
    return mandal_to_feature(row.iloc[0])

# ── Projections items ─────────────────────────────────────────────────────────
@app.get("/collections/drought-projections/items", tags=["Projections"])
def projection_items(
    district: str  = Query(None, description="Filter by district"),
    scenario: str  = Query(None, description="SSP2-4.5 or SSP5-8.5"),
    year:     int  = Query(None, ge=2026, le=2040, description="Target year"),
    limit:    int  = Query(100, ge=1, le=1000),
):
    df = proj.copy()
    if district: df = df[df["district"].str.lower() == district.lower()]
    if scenario: df = df[df["scenario"] == scenario]
    if year:     df = df[df["year"] == year]
    if df.empty:
        raise HTTPException(404, "No projections match the query")

    features = []
    for _, row in df.iloc[:limit].iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(row["longitude"],5), round(row["latitude"],5)]},
            "properties": {
                "district":            row["district"],
                "year":                int(row["year"]),
                "scenario":            row["scenario"],
                "drought_probability": round(row["drought_probability"], 4),
                "proj_rainfall_mm":    round(row["proj_rainfall_mm"],    1),
                "proj_temp_c":         round(row["proj_temp_c"],         2),
                "proj_spi3":           round(row["proj_spi3"],           3),
                "vulnerability":       round(row["vulnerability"],       3),
                "cdsi_mean":           round(row["cdsi_mean"],           4),
                "data_source":         row["data_source"],
                "phenomenon_time":     f"{int(row['year'])}-01-01",
                "result_quality":      "20-member ensemble · IPCC AR6 WG1 South Asia",
            }
        })

    return {
        "type": "FeatureCollection",
        "timeStamp": NOW(),
        "numberReturned": len(features),
        "features": features,
        "links": [{"href": "/collections/drought-projections/items", "rel": "self"}],
        "_anvikshana_meta": {
            "projection_horizon": "2026–2040",
            "scenarios":          ["SSP2-4.5", "SSP5-8.5"],
            "ensemble_members":   20,
            "base_model":         "Random Forest Classifier · ROC-AUC 0.974",
            "climate_forcing":    "IPCC AR6 WG1 South Asia regional change factors",
            "standards":          ["OGC CSAPI", "OGC API-EDR"],
        }
    }

# ── Single projection ─────────────────────────────────────────────────────────
@app.get("/collections/drought-projections/items/{district}/{scenario}/{year}", tags=["Projections"])
def projection_item(district: str, scenario: str, year: int):
    row = proj[
        (proj["district"].str.lower() == district.lower()) &
        (proj["scenario"] == scenario) &
        (proj["year"] == year)
    ]
    if row.empty:
        raise HTTPException(404, f"No projection for {district} / {scenario} / {year}")
    r = row.iloc[0]
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(r["longitude"],5), round(r["latitude"],5)]},
        "properties": {
            "district": r["district"], "year": int(r["year"]), "scenario": r["scenario"],
            "drought_probability": round(r["drought_probability"], 4),
            "proj_rainfall_mm": round(r["proj_rainfall_mm"], 1),
            "proj_temp_c": round(r["proj_temp_c"], 2),
            "proj_spi3": round(r["proj_spi3"], 3),
            "vulnerability": round(r["vulnerability"], 3),
            "data_source": r["data_source"],
        }
    }

# ── OGC EDR Position Query ────────────────────────────────────────────────────
@app.get("/position", tags=["OGC EDR"])
def position_query(
    coords: str = Query(..., description="WKT point: POINT(lon lat) e.g. POINT(79.5 17.8)"),
    parameter_name: str = Query("cdsi", description="Parameter: cdsi | ndvi_mean | sm_mean"),
    datetime_param: str = Query(None, alias="datetime", description="ISO date e.g. 2025-09-14"),
):
    """OGC API-EDR /position endpoint — nearest mandal to given coordinates."""
    try:
        coords_clean = coords.replace("POINT(","").replace(")","").strip()
        lon, lat = map(float, coords_clean.split())
    except Exception:
        raise HTTPException(400, "coords must be WKT: POINT(lon lat)")

    df = latest.copy()
    df["_dist"] = ((df["longitude"] - lon)**2 + (df["latitude"] - lat)**2)**0.5
    nearest = df.nsmallest(1, "_dist").iloc[0]

    if parameter_name not in ["cdsi","ndvi_mean","sm_mean","drought_class"]:
        raise HTTPException(400, f"parameter_name '{parameter_name}' not supported")

    return {
        "type": "Coverage",
        "domain": {
            "type":  "Domain",
            "axes":  {"x": {"values": [nearest["longitude"]]}, "y": {"values": [nearest["latitude"]]}},
            "referencing": [{"coordinates": ["x","y"], "system": {"type": "GeographicCRS", "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}}]
        },
        "parameters": {
            parameter_name: {
                "type":              "Parameter",
                "description":       f"Anvīkṣaṇa {parameter_name}",
                "observedProperty":  {"label": {"en": parameter_name}},
                "unit":              {"label": {"en": "dimensionless"}},
            }
        },
        "ranges": {
            parameter_name: {
                "type":   "NdArray",
                "dataType": "float",
                "values": [round(float(nearest[parameter_name]), 4) if parameter_name != "drought_class" else nearest[parameter_name]]
            }
        },
        "_nearest_mandal": {
            "uid":           nearest["uid"],
            "mandal":        nearest["mandal"],
            "district":      nearest["district"],
            "distance_deg":  round(float(nearest["_dist"]), 4),
            "observation_date": str(latest_date)[:10],
        }
    }

# ── Health / Data Provenance ──────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status":   "operational",
        "api":      "Anvīkṣaṇa OGC CSAPI v1.0",
        "timestamp": NOW(),
        "data_provenance": {
            "primary_source":     "DiCRA / UNDP India — Digital Public Good",
            "dicra_records":      487243,
            "mandal_polygons":    int(latest["mandal"].nunique()),
            "districts":          int(latest["district"].nunique()),
            "observation_dates":  int(merged["date"].nunique()),
            "latest_observation": str(latest_date)[:10],
            "climate_baseline":   "NASA POWER GMAO 2000–2024",
            "projection_horizon": "2026–2040",
        },
        "model_performance": {
            "roc_auc":    "0.974 ± 0.004",
            "f1_score":   "0.801 ± 0.032",
            "brier_score": 0.058,
            "validation": "Spatial GroupKFold k=5 · district hold-out",
            "historical_detection": "5/5 declared drought years (2002–2019)",
        },
        "standards_conformance": [
            "OGC API - Features 1.0",
            "OGC API - EDR 1.1",
            "OGC CSAPI (Climate Stack API) draft",
            "W3C WoT (Web of Things)",
            "ISO 19179 (in review)",
        ],
        "organisation": {
            "name":    "Aganitha Space Technologies Pvt. Ltd.",
            "contact": "nsnarayanam@aganithaspace.com",
            "dpiit":   "DIPP162965",
            "standards_bodies": ["OGC CSAPI SWG", "IEEE GRSS P4011", "W3C WoT WG"],
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ogc_api:app", host="0.0.0.0", port=8502, reload=True)
