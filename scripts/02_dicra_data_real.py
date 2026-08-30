"""
NSCIC Prototype - Script 02 (real data): DiCRA ingestion
=========================================================
Builds telangana_h3_climate.parquet from OBSERVED data only.

Replaces generate_synthetic_dicra_data() in 02_dicra_data.py, which
populated ndvi / soil_moisture / lst_celsius / rainfall_mm from seasonal
curves plus np.random noise. No value written by this script is simulated.

DATA PROVENANCE
---------------
  ndvi_*            data/geojson_ndvi/DD-MM-YYYY.geojson
                    DiCRA / UNDP India zonal statistics, 592 mandal polygons,
                    23 biweekly dates in 2025.
  sm_mean           data/geojson_ndvi/mandal_ndvi_sm_merged.csv
                    DiCRA soil moisture, mandal level, same 23 dates.
  rain_clim_mm      data/geojson_ndvi/nasa_power_telangana.csv
  temp_clim_c       NASA POWER / GMAO, district level, 2000-2024.
                    Aggregated to a per-district monthly CLIMATOLOGY.
                    NOT observed 2025 rainfall. See LIMITATION 1.

  IMD is not read by this script. The IMD files in data/geojson_ndvi/ serve
  the dashboard's live-alerts page and the external comparison only. They
  enter no column of this parquet.

LIMITATIONS carried into the parquet
------------------------------------
  1. TEMPORAL MISMATCH. Mandal NDVI and soil moisture are 2025. NASA POWER
     ends 2024. The rainfall and temperature columns are therefore long-term
     monthly normals joined by (district, month), not contemporaneous
     observations. They describe the district's usual conditions in that
     month, not what actually fell in 2025. Any Methods section must say so.

  2. SINGLE-SEASON VCI. VCI needs a multi-year NDVI min/max envelope. Only
     2025 is available at mandal level here, so vci_2025_only is computed
     against the within-2025 range and measures position in the annual cycle,
     not interannual drought. It is emitted for continuity with the dashboard
     and MUST NOT be used as a model feature or a label.

  3. SPATIAL MISMATCH. Rainfall and temperature are district-level, broadcast
     to every mandal in that district. All mandals of a district share a
     value. Do not read mandal-scale rainfall signal into it.

  4. NO LABEL. is_drought is intentionally empty. See below.

WHY THERE IS NO LABEL COLUMN
----------------------------
The previous pipeline set is_drought = (cdsi > 1.0) where cdsi was an
arithmetic combination of the same rainfall, NDVI and soil-moisture columns
that were then fed to the classifier as features. The target was a
deterministic function of the inputs. That is label leakage and it is the
most likely explanation for ROC-AUC 0.974.

Every label-shaped column shipped in the CSVs has the same defect:
vci_class, drought_class, drought_risk_ndvi and drought_freq_pct are all
transforms of NDVI, soil moisture or SPI. None is an independent label.

The label must come from outside the feature set. The intended source is the
Telangana government drought declarations already listed by year in
04_benchmark.py, which need the per-district (ideally per-mandal) name lists
from the Revenue Department GOs for 2015, 2017, 2018 and 2019.

Populate via:  python 02_dicra_data_real.py --labels declarations.csv
    expected columns: year, district [, mandal], declared
Until then is_drought stays NaN and 03_drought_model.py will refuse to train.

Usage:  python 02_dicra_data_real.py [--labels FILE] [--resolution 7]
Output: data/processed/telangana_h3_climate.parquet
        data/processed/provenance.json
"""

import os
import re
import json
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, '..', 'data', 'geojson_ndvi')
PROCESSED_DIR = os.path.join(HERE, '..', 'data', 'processed')

# Columns that are arithmetic transforms of other columns in this table.
# Feeding these to a classifier alongside a threshold-derived label is how
# the previous leakage happened. 03 should exclude them explicitly.
DERIVED_UNSAFE = ['vci_2025_only', 'ndvi_norm_2025', 'sm_norm_2025']


# ─────────────────────────────────────────────────────────────────────────────
# 1. NDVI from DiCRA GeoJSON
# ─────────────────────────────────────────────────────────────────────────────

def parse_geojson_date(path):
    """Filenames are DD-MM-YYYY.geojson."""
    stem = os.path.basename(path).replace('.geojson', '')
    if not re.fullmatch(r'\d{2}-\d{2}-\d{4}', stem):
        raise ValueError(f"Unexpected geojson filename: {stem}")
    return datetime.strptime(stem, '%d-%m-%Y').date()


def load_ndvi():
    """Read every dated DiCRA GeoJSON into one mandal x date frame."""
    paths = sorted(glob.glob(os.path.join(RAW_DIR, '*.geojson')))
    if not paths:
        raise FileNotFoundError(f"No GeoJSON files in {RAW_DIR}")

    rows = []
    for path in paths:
        obs_date = parse_geojson_date(path)
        with open(path) as fh:
            gj = json.load(fh)

        for feat in gj['features']:
            props = feat['properties']
            zs = props.get('zonalstat') or {}
            if zs.get('mean') is None:
                continue
            lon, lat = props.get('centroid', [np.nan, np.nan])
            rows.append({
                'date': obs_date,
                'uid': props.get('uid'),
                'mandal': props.get('mandal_name'),
                'district': props.get('district_name'),
                'area_km2': props.get('area'),
                'longitude': lon,
                'latitude': lat,
                'ndvi_mean': zs.get('mean'),
                'ndvi_min': zs.get('min'),
                'ndvi_max': zs.get('max'),
                'ndvi_median': zs.get('median'),
                'ndvi_pixels': zs.get('count'),
            })

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  NDVI: {len(df):,} rows | {df['uid'].nunique()} mandals "
          f"| {df['date'].nunique()} dates | {df['district'].nunique()} districts")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Soil moisture from the DiCRA merged CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_soil_moisture():
    """Take ONLY the observed SM columns. Ignore cdsi / drought_class."""
    path = os.path.join(RAW_DIR, 'mandal_ndvi_sm_merged.csv')
    sm = pd.read_csv(path, usecols=['date', 'uid', 'sm_mean', 'sm_min', 'sm_max'])
    sm['date'] = pd.to_datetime(sm['date'])
    sm = sm.dropna(subset=['sm_mean'])
    print(f"  Soil moisture: {len(sm):,} rows | {sm['uid'].nunique()} mandals")
    return sm


# ─────────────────────────────────────────────────────────────────────────────
# 3. NASA POWER monthly climatology
# ─────────────────────────────────────────────────────────────────────────────

def load_nasa_climatology():
    """
    Collapse 2000-2024 NASA POWER to a per-district, per-month normal.

    This is a CLIMATOLOGY, not an observation. Column names carry _clim_
    so no downstream reader can mistake it for 2025 rainfall.
    """
    path = os.path.join(RAW_DIR, 'nasa_power_telangana.csv')
    np_df = pd.read_csv(path)

    clim = (np_df.groupby(['district', 'month'])
                 .agg(rain_clim_mm=('rainfall_mm', 'mean'),
                      rain_clim_sd=('rainfall_mm', 'std'),
                      temp_clim_c=('temp_mean_c', 'mean'),
                      tmax_clim_c=('temp_max_c', 'mean'),
                      humidity_clim_pct=('humidity_pct', 'mean'),
                      clim_n_years=('year', 'nunique'))
                 .reset_index())

    print(f"  NASA POWER climatology: {len(clim)} district-month normals "
          f"from {np_df['year'].min()}-{np_df['year'].max()}")
    return clim


# ─────────────────────────────────────────────────────────────────────────────
# 4. Assemble
# ─────────────────────────────────────────────────────────────────────────────

def normalise_district(s):
    return (s.astype(str).str.strip().str.lower()
             .str.replace(r'\s+', ' ', regex=True))


def assemble(ndvi, sm, clim):
    df = ndvi.merge(sm, on=['date', 'uid'], how='left')
    n_missing_sm = df['sm_mean'].isna().sum()
    if n_missing_sm:
        print(f"  ! {n_missing_sm:,} rows without soil moisture "
              f"({n_missing_sm / len(df):.1%})")

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['doy'] = df['date'].dt.dayofyear

    df['_dkey'] = normalise_district(df['district'])
    clim = clim.copy()
    clim['_dkey'] = normalise_district(clim['district'])

    unmatched = set(df['_dkey']) - set(clim['_dkey'])
    if unmatched:
        print(f"  ! {len(unmatched)} districts absent from NASA POWER: "
              f"{sorted(unmatched)}")

    df = df.merge(clim.drop(columns=['district']), on=['_dkey', 'month'], how='left')
    df = df.drop(columns=['_dkey'])

    # Season, from the calendar only.
    df['season'] = df['month'].map(
        lambda m: 'winter' if m in (12, 1, 2)
        else 'pre_monsoon' if m in (3, 4, 5)
        else 'monsoon' if m in (6, 7, 8, 9)
        else 'post_monsoon')

    # Within-2025 normalisations. Emitted for dashboard continuity only.
    # See LIMITATION 2. Listed in DERIVED_UNSAFE.
    g = df.groupby('uid')['ndvi_mean']
    lo, hi = g.transform('min'), g.transform('max')
    rng = (hi - lo).replace(0, np.nan)
    df['ndvi_norm_2025'] = (df['ndvi_mean'] - lo) / rng
    df['vci_2025_only'] = df['ndvi_norm_2025'] * 100

    gs = df.groupby('uid')['sm_mean']
    slo, shi = gs.transform('min'), gs.transform('max')
    srng = (shi - slo).replace(0, np.nan)
    df['sm_norm_2025'] = (df['sm_mean'] - slo) / srng

    # Deliberately empty. See module docstring.
    df['is_drought'] = np.nan

    return df.sort_values(['uid', 'date']).reset_index(drop=True)


def add_h3(df, resolution=7):
    if not HAS_H3:
        print("  ! h3 not installed, skipping H3 index (pip install h3)")
        df['h3_index'] = None
        return df

    cell = getattr(h3, 'latlng_to_cell', None) or h3.geo_to_h3
    df['h3_index'] = [
        cell(la, lo, resolution) if pd.notna(la) and pd.notna(lo) else None
        for la, lo in zip(df['latitude'], df['longitude'])
    ]
    print(f"  H3 res-{resolution}: {df['h3_index'].nunique():,} distinct cells")
    return df


def attach_labels(df, labels_path):
    """
    Join external drought declarations.

    Expected CSV: year, district [, mandal], declared
    Mandal-level rows win over district-level rows for the same year.
    """
    lab = pd.read_csv(labels_path)
    required = {'year', 'district', 'declared'}
    if not required.issubset(lab.columns):
        raise ValueError(f"{labels_path} needs columns {required}")

    lab['_dkey'] = normalise_district(lab['district'])
    df['_dkey'] = normalise_district(df['district'])

    if 'mandal' in lab.columns and lab['mandal'].notna().any():
        lab['_mkey'] = lab['mandal'].astype(str).str.strip().str.lower()
        df['_mkey'] = df['mandal'].astype(str).str.strip().str.lower()
        mand = lab.dropna(subset=['mandal'])[['year', '_dkey', '_mkey', 'declared']]
        df = df.merge(mand.rename(columns={'declared': '_lab_m'}),
                      on=['year', '_dkey', '_mkey'], how='left')
    else:
        df['_lab_m'] = np.nan

    dist = (lab[lab.get('mandal').isna()] if 'mandal' in lab.columns else lab)
    dist = dist[['year', '_dkey', 'declared']].drop_duplicates()
    df = df.merge(dist.rename(columns={'declared': '_lab_d'}),
                  on=['year', '_dkey'], how='left')

    df['is_drought'] = df['_lab_m'].combine_first(df['_lab_d']).astype('float')
    df = df.drop(columns=[c for c in ['_dkey', '_mkey', '_lab_m', '_lab_d']
                          if c in df.columns])

    n = df['is_drought'].notna().sum()
    print(f"  Labels: {n:,}/{len(df):,} rows ({n / len(df):.1%}) | "
          f"positive rate {df['is_drought'].mean():.1%}")
    if n < len(df):
        print("  ! Unlabelled rows must be dropped before training, not imputed.")
    return df


def write_provenance(df, out_path, labels_path):
    prov = {
        'generated_utc': datetime.now().astimezone().isoformat(),
        'script': os.path.basename(__file__),
        'synthetic_data_used': False,
        'sources': {
            'ndvi': 'DiCRA / UNDP India zonal statistics, 23 dates 2025, mandal level',
            'soil_moisture': 'DiCRA soil moisture, mandal level, 2025',
            'rainfall_temperature': 'NASA POWER / GMAO 2000-2024, district monthly climatology',
            'imd': 'not ingested; dashboard display and external comparison only',
        },
        'rows': int(len(df)),
        'mandals': int(df['uid'].nunique()),
        'districts': int(df['district'].nunique()),
        'dates': int(df['date'].nunique()),
        'date_range': [str(df['date'].min().date()), str(df['date'].max().date())],
        'label_source': labels_path or None,
        'labelled_rows': int(df['is_drought'].notna().sum()),
        'derived_columns_unsafe_as_features': DERIVED_UNSAFE,
        'limitations': [
            'Rainfall and temperature are 2000-2024 monthly climatology joined by '
            '(district, month), not observed 2025 values.',
            'Rainfall and temperature are district level, broadcast to all mandals '
            'in the district.',
            'vci_2025_only uses a within-2025 NDVI envelope and does not measure '
            'interannual drought.',
        ],
    }
    with open(out_path, 'w') as fh:
        json.dump(prov, fh, indent=2)
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=None,
                    help='CSV of drought declarations: year, district[, mandal], declared')
    ap.add_argument('--resolution', type=int, default=7, help='H3 resolution')
    args = ap.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("=" * 66)
    print("DiCRA ingestion - observed data only")
    print("=" * 66)

    ndvi = load_ndvi()
    sm = load_soil_moisture()
    clim = load_nasa_climatology()

    print("\nAssembling...")
    df = assemble(ndvi, sm, clim)
    df = add_h3(df, args.resolution)

    if args.labels:
        print("\nAttaching labels...")
        df = attach_labels(df, args.labels)
    else:
        print("\n  No --labels supplied. is_drought left empty.")

    out = os.path.join(PROCESSED_DIR, 'telangana_h3_climate.parquet')
    df.to_parquet(out, index=False)
    prov = write_provenance(df, os.path.join(PROCESSED_DIR, 'provenance.json'),
                            args.labels)

    print("\n" + "=" * 66)
    print(f"Wrote {out}")
    print(f"  {prov['rows']:,} rows | {prov['mandals']} mandals | "
          f"{prov['districts']} districts | {prov['dates']} dates")
    print(f"  Labelled: {prov['labelled_rows']:,}")
    print("=" * 66)
    if not prov['labelled_rows']:
        print("\nNEXT: source Telangana Revenue Dept drought declaration GOs")
        print("      (2015, 2017, 2018, 2019), build declarations.csv, re-run")
        print("      with --labels. Do not train until then.")


if __name__ == '__main__':
    main()
