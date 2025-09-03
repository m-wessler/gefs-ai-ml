#!/usr/bin/env python3

"""
Debug script to analyze data alignment issues when aggregating across forecast hours.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# Configuration matching main script
BASE_PATH = "N:/data/gefs-ml/"
STATION_IDS = ['KSLC']
FORECAST_HOURS = ['f024', 'f048']
TARGET_VARIABLE = 'tmax'

def load_and_examine_data():
    """Load data and examine alignment issues."""
    print("=== DEBUGGING DATA ALIGNMENT ISSUES ===")
    
    # Load forecast data
    forecast_dfs = []
    for fhour in FORECAST_HOURS:
        data_dir = Path(BASE_PATH) / 'forecast' / fhour
        for station_id in STATION_IDS:
            file_path = data_dir / f"{station_id}_2020_2025_{fhour}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['sid'] = station_id
                df['forecast_hour'] = fhour
                forecast_dfs.append(df)
                print(f"Loaded {fhour} forecast: {df.shape}")
    
    # Load NBM data
    nbm_dfs = []
    for fhour in FORECAST_HOURS:
        data_dir = Path(BASE_PATH) / 'nbm' / fhour
        for station_id in STATION_IDS:
            file_path = data_dir / f"{station_id}_2020_2025_{fhour}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['sid'] = station_id
                df['forecast_hour'] = fhour
                nbm_dfs.append(df)
                print(f"Loaded {fhour} NBM: {df.shape}")
    
    # Combine data
    forecast_df = pd.concat(forecast_dfs, ignore_index=True) if forecast_dfs else pd.DataFrame()
    nbm_df = pd.concat(nbm_dfs, ignore_index=True) if nbm_dfs else pd.DataFrame()
    
    # Load URMA data
    urma_path = Path(BASE_PATH) / 'urma' / 'WR_2020_2025.urma.parquet'
    urma_df = pd.read_parquet(urma_path, engine='pyarrow')
    urma_df = urma_df.loc[urma_df.index.get_level_values(1).isin(STATION_IDS)]
    
    print(f"\nCombined forecast data: {forecast_df.shape}")
    print(f"Combined NBM data: {nbm_df.shape}")
    print(f"URMA data: {urma_df.shape}")
    
    # Examine data structure
    print("\n=== EXAMINING DATA STRUCTURE ===")
    
    # Convert datetime columns
    forecast_df['valid_datetime'] = pd.to_datetime(forecast_df['valid_datetime'])
    nbm_df['valid_datetime'] = pd.to_datetime(nbm_df['valid_datetime'])
    
    # Look at sample data
    print("\nSample forecast data:")
    sample_forecast = forecast_df.head(10)[['valid_datetime', 'sid', 'forecast_hour', 'tmax_2m']]
    print(sample_forecast)
    
    print("\nSample NBM data:")
    sample_nbm = nbm_df.head(10)[['valid_datetime', 'sid', 'forecast_hour', 'tmax']]
    print(sample_nbm)
    
    print("\nSample URMA data:")
    sample_urma = urma_df.head(10)
    print(sample_urma)
    
    # Check temporal alignment
    print("\n=== CHECKING TEMPORAL ALIGNMENT ===")
    
    # Group by valid_datetime to see how many forecast hours per time
    forecast_times = forecast_df.groupby(['valid_datetime', 'sid']).size()
    nbm_times = nbm_df.groupby(['valid_datetime', 'sid']).size()
    
    print(f"Forecast entries per (time, station): min={forecast_times.min()}, max={forecast_times.max()}, mean={forecast_times.mean():.2f}")
    print(f"NBM entries per (time, station): min={nbm_times.min()}, max={nbm_times.max()}, mean={nbm_times.mean():.2f}")
    
    # Check for specific alignment issues
    print("\n=== CHECKING FOR ALIGNMENT ISSUES ===")
    
    # Find a sample URMA time and see what forecast/NBM data matches
    urma_times = urma_df.index.get_level_values('valid_time')
    sample_urma_time = urma_times[100]  # Pick a random URMA time
    station = STATION_IDS[0]
    
    print(f"Sample URMA time: {sample_urma_time}")
    print(f"Station: {station}")
    
    # Find matching forecasts in 12-hour window
    window_start = sample_urma_time - timedelta(hours=12)
    window_end = sample_urma_time
    
    forecast_matches = forecast_df[
        (forecast_df['valid_datetime'] >= window_start) &
        (forecast_df['valid_datetime'] <= window_end) &
        (forecast_df['sid'] == station)
    ]
    
    nbm_matches = nbm_df[
        (nbm_df['valid_datetime'] >= window_start) &
        (nbm_df['valid_datetime'] <= window_end) &
        (nbm_df['sid'] == station)
    ]
    
    print(f"\nForecast matches in window:")
    print(forecast_matches[['valid_datetime', 'forecast_hour', 'tmax_2m']])
    
    print(f"\nNBM matches in window:")
    print(nbm_matches[['valid_datetime', 'forecast_hour', 'tmax']])
    
    # Check if the values are properly aligned
    if not forecast_matches.empty and not nbm_matches.empty:
        print("\n=== CHECKING VALUE ALIGNMENT ===")
        
        # Merge forecast and NBM on valid_datetime and forecast_hour
        merged = pd.merge(
            forecast_matches, 
            nbm_matches, 
            on=['valid_datetime', 'sid', 'forecast_hour'],
            suffixes=('_forecast', '_nbm')
        )
        
        if not merged.empty:
            print("Merged forecast/NBM data:")
            print(merged[['valid_datetime', 'forecast_hour', 'tmax_2m', 'tmax']])
            
            # Check correlation
            correlation = merged['tmax_2m'].corr(merged['tmax'])
            print(f"GEFS-NBM correlation: {correlation:.3f}")
        else:
            print("No matching forecast/NBM pairs found!")
    
    return forecast_df, nbm_df, urma_df

if __name__ == "__main__":
    load_and_examine_data()
