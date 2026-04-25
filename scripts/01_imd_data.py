"""
NSCIC Prototype — Script 1: IMD Data Acquisition
Downloads IMD gridded rainfall and temperature data for Telangana
using the imdlib Python library, then processes to district-level statistics.

Usage: python 01_imd_data.py
Output: data/imd/telangana_rainfall.csv, data/imd/telangana_temperature.csv
"""

import imdlib as imd
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Telangana bounding box
TEL_LAT_MIN, TEL_LAT_MAX = 15.8, 19.9
TEL_LON_MIN, TEL_LON_MAX = 77.2, 81.3

# Output directory
IMD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'imd')
os.makedirs(IMD_DIR, exist_ok=True)

def download_imd_rainfall(start_year=2000, end_year=2024):
    """Download IMD 0.25° gridded daily rainfall data."""
    print(f"Downloading IMD rainfall data ({start_year}-{end_year})...")
    try:
        # imdlib downloads to current directory by default
        data = imd.open_data('rain', start_year, end_year, 'yearwise')
        
        # Get the xarray dataset
        ds = data.get_xarray()
        
        # Subset to Telangana
        ds_tel = ds.sel(
            lat=slice(TEL_LAT_MIN, TEL_LAT_MAX),
            lon=slice(TEL_LON_MIN, TEL_LON_MAX)
        )
        
        # Save as NetCDF
        outpath = os.path.join(IMD_DIR, 'telangana_rainfall.nc')
        ds_tel.to_netcdf(outpath)
        print(f"Saved Telangana rainfall to {outpath}")
        
        # Also compute monthly district-average for quick analysis
        monthly = ds_tel['rain'].resample(time='M').sum()
        df = monthly.mean(dim=['lat', 'lon']).to_dataframe().reset_index()
        df.columns = ['date', 'avg_rainfall_mm']
        csv_path = os.path.join(IMD_DIR, 'telangana_monthly_rainfall.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved monthly averages to {csv_path}")
        
        return ds_tel
    except Exception as e:
        print(f"imdlib download failed: {e}")
        print("Falling back to manual download instructions...")
        print_manual_instructions()
        return None

def download_imd_temperature(start_year=2000, end_year=2024):
    """Download IMD 1° gridded daily temperature data."""
    print(f"Downloading IMD temperature data ({start_year}-{end_year})...")
    try:
        # Max temperature
        data_max = imd.open_data('tmax', start_year, end_year, 'yearwise')
        ds_max = data_max.get_xarray()
        ds_max_tel = ds_max.sel(
            lat=slice(TEL_LAT_MIN, TEL_LAT_MAX),
            lon=slice(TEL_LON_MIN, TEL_LON_MAX)
        )
        
        # Min temperature
        data_min = imd.open_data('tmin', start_year, end_year, 'yearwise')
        ds_min = data_min.get_xarray()
        ds_min_tel = ds_min.sel(
            lat=slice(TEL_LAT_MIN, TEL_LAT_MAX),
            lon=slice(TEL_LON_MIN, TEL_LON_MAX)
        )
        
        # Save
        ds_max_tel.to_netcdf(os.path.join(IMD_DIR, 'telangana_tmax.nc'))
        ds_min_tel.to_netcdf(os.path.join(IMD_DIR, 'telangana_tmin.nc'))
        print("Saved temperature data for Telangana")
        
        return ds_max_tel, ds_min_tel
    except Exception as e:
        print(f"Temperature download failed: {e}")
        return None, None

def compute_spi(rainfall_series, scale=3):
    """
    Compute Standardised Precipitation Index (SPI) at given timescale.
    SPI is the core drought indicator used by IMD and WMO.
    
    Args:
        rainfall_series: Monthly rainfall time series (pd.Series)
        scale: Accumulation period in months (3, 6, 12)
    
    Returns:
        SPI values (pd.Series)
    """
    from scipy import stats
    
    # Rolling sum over the accumulation period
    rolling = rainfall_series.rolling(window=scale).sum()
    
    # Fit gamma distribution to positive values
    valid = rolling.dropna()
    valid_pos = valid[valid > 0]
    
    if len(valid_pos) < 30:
        print(f"Warning: insufficient data for SPI-{scale} computation")
        return pd.Series(np.nan, index=rainfall_series.index)
    
    # Gamma fit
    shape, loc, scale_param = stats.gamma.fit(valid_pos, floc=0)
    
    # Compute CDF
    cdf = stats.gamma.cdf(rolling, shape, loc=0, scale=scale_param)
    
    # Handle zero precipitation
    q = len(valid[valid == 0]) / len(valid)
    cdf = q + (1 - q) * cdf
    
    # Transform to standard normal
    spi = stats.norm.ppf(cdf)
    spi = pd.Series(spi, index=rainfall_series.index)
    
    # Clip extreme values
    spi = spi.clip(-3, 3)
    
    return spi

def compute_district_spi(rainfall_nc_path):
    """Compute SPI at multiple timescales for Telangana average."""
    import xarray as xr
    
    ds = xr.open_dataset(rainfall_nc_path)
    
    # Spatial average monthly rainfall
    monthly = ds['rain'].resample(time='M').sum().mean(dim=['lat', 'lon'])
    df = monthly.to_dataframe().reset_index()
    df.columns = ['date', 'rainfall_mm']
    
    # Compute SPI at 3, 6, 12 month scales
    for scale in [3, 6, 12]:
        df[f'spi_{scale}'] = compute_spi(df['rainfall_mm'], scale=scale)
    
    # Drought classification
    df['drought_category'] = pd.cut(
        df['spi_3'],
        bins=[-np.inf, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, np.inf],
        labels=['Exceptional', 'Extreme', 'Severe', 'Normal',
                'Moderately Wet', 'Very Wet', 'Exceptionally Wet']
    )
    
    csv_path = os.path.join(IMD_DIR, 'telangana_spi.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved SPI data to {csv_path}")
    
    return df

def print_manual_instructions():
    """Print manual download instructions if imdlib fails."""
    print("""
    ============================================================
    MANUAL DOWNLOAD INSTRUCTIONS FOR IMD DATA
    ============================================================
    
    1. RAINFALL (0.25° x 0.25°, daily, 1901-2024):
       Go to: https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
       Download yearly NetCDF files for 2000-2024
       Save to: data/imd/
    
    2. MAX TEMPERATURE (1° x 1°, daily, 1951-2024):
       Go to: https://cdsp.imdpune.gov.in/home_gridded_data.php
       Download yearly files
       Save to: data/imd/
    
    3. Alternative — use imdlib from command line:
       pip install imdlib
       python -c "import imdlib; imdlib.get_data('rain', 2000, 2024, 'yearwise')"
    
    4. Alternative — use imddata CLI tool:
       pip install imddata
       imddata download rain --start 2000 --end 2024
    ============================================================
    """)

if __name__ == '__main__':
    print("=" * 60)
    print("NSCIC Stage 2 — IMD Data Acquisition Pipeline")
    print("Geography: Telangana | Hazard: Agricultural Drought")
    print("=" * 60)
    
    # Step 1: Download rainfall
    ds_rain = download_imd_rainfall(2000, 2024)
    
    # Step 2: Download temperature
    ds_tmax, ds_tmin = download_imd_temperature(2000, 2024)
    
    # Step 3: Compute SPI
    if ds_rain is not None:
        rain_nc = os.path.join(IMD_DIR, 'telangana_rainfall.nc')
        if os.path.exists(rain_nc):
            spi_df = compute_district_spi(rain_nc)
            print("\nSPI Summary (last 12 months):")
            print(spi_df.tail(12)[['date', 'rainfall_mm', 'spi_3', 'spi_6', 'drought_category']])
    
    print("\nIMD data pipeline complete!")
