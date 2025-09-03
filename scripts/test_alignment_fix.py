#!/usr/bin/env python3

"""
Test script to verify the improved data alignment works correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# Test configuration
BASE_PATH = "N:/data/gefs-ml/"
STATION_IDS = ['KSLC']
FORECAST_HOURS = ['f024', 'f048']
TARGET_VARIABLE = 'tmax'

def quick_alignment_test():
    """Quick test of the improved alignment logic."""
    print("=== TESTING IMPROVED DATA ALIGNMENT ===")
    
    # Load a small sample of data for testing
    forecast_dfs = []
    nbm_dfs = []
    
    print("Loading test data...")
    for fhour in FORECAST_HOURS:
        # Load forecast data
        forecast_dir = Path(BASE_PATH) / 'forecast' / fhour
        forecast_file = forecast_dir / f"{STATION_IDS[0]}_2020_2025_{fhour}.csv"
        if forecast_file.exists():
            df = pd.read_csv(forecast_file)
            df['sid'] = STATION_IDS[0]
            df['forecast_hour'] = fhour
            # Take only recent data for quick test
            df = df.tail(100)
            forecast_dfs.append(df)
            print(f"  Loaded {len(df)} {fhour} forecast records")
        
        # Load NBM data
        nbm_dir = Path(BASE_PATH) / 'nbm' / fhour
        nbm_file = nbm_dir / f"{STATION_IDS[0]}_2020_2025_{fhour}.csv"
        if nbm_file.exists():
            df = pd.read_csv(nbm_file)
            df['sid'] = STATION_IDS[0]
            df['forecast_hour'] = fhour
            # Take only recent data for quick test
            df = df.tail(100)
            nbm_dfs.append(df)
            print(f"  Loaded {len(df)} {fhour} NBM records")
    
    if not forecast_dfs or not nbm_dfs:
        print("Error: Could not load test data")
        return
    
    # Combine data
    forecast_df = pd.concat(forecast_dfs, ignore_index=True)
    nbm_df = pd.concat(nbm_dfs, ignore_index=True)
    
    # Convert datetime and set index
    forecast_df['valid_datetime'] = pd.to_datetime(forecast_df['valid_datetime'])
    nbm_df['valid_datetime'] = pd.to_datetime(nbm_df['valid_datetime'])
    
    forecast_df = forecast_df.set_index(['valid_datetime', 'sid'])
    nbm_df = nbm_df.set_index(['valid_datetime', 'sid'])
    
    print(f"Combined forecast data: {forecast_df.shape}")
    print(f"Combined NBM data: {nbm_df.shape}")
    
    # Load small URMA sample
    urma_path = Path(BASE_PATH) / 'urma' / 'WR_2020_2025.urma.parquet'
    urma_df = pd.read_parquet(urma_path, engine='pyarrow')
    urma_df = urma_df.loc[urma_df.index.get_level_values(1).isin(STATION_IDS)]
    
    # Take recent URMA data
    urma_times = urma_df.index.get_level_values('valid_time')
    recent_times = urma_times[-50:]  # Last 50 times
    urma_df = urma_df.loc[urma_df.index.get_level_values('valid_time').isin(recent_times)]
    
    print(f"Test URMA data: {urma_df.shape}")
    
    # Prepare subsets (simplified)
    forecast_cols = ['tmax_2m']
    nbm_cols = ['tmax']
    
    forecast_subset = forecast_df[forecast_cols].dropna()
    nbm_subset = nbm_df[nbm_cols].dropna()
    urma_subset = urma_df[['tmax']].dropna()
    
    # Add prefixes
    forecast_subset.columns = ['gefs_' + col for col in forecast_subset.columns]
    nbm_subset.columns = ['nbm_' + col for col in nbm_subset.columns]
    
    # Standardize URMA
    urma_subset = urma_subset.reset_index().rename(
        columns={'tmax': 'tmax', 'valid_time': 'valid_datetime', 'station_id': 'sid'}
    ).set_index(['valid_datetime', 'sid'])
    urma_subset.columns = ['urma_' + col for col in urma_subset.columns]
    urma_subset['urma_tmax'] -= 273.15  # K to C
    
    print("\nTesting improved time matching...")
    
    # Test the improved matching logic
    matched_data = test_improved_time_matching(forecast_subset, nbm_subset, urma_subset)
    
    if not matched_data.empty:
        print(f"Successfully created {len(matched_data)} matched records")
        
        # Test alignment verification
        print("\nTesting alignment verification...")
        target_config = {
            'forecast_col': 'tmax_2m',
            'urma_obs_col': 'urma_tmax'
        }
        test_alignment_verification(matched_data, target_config)
        
        print("\n=== TEST COMPLETED SUCCESSFULLY ===")
    else:
        print("Error: No matched data created")

def test_improved_time_matching(forecast_subset, nbm_subset, urma_subset):
    """Test the improved time matching logic."""
    urma_times = urma_subset.index.get_level_values('valid_datetime').unique()
    stations = urma_subset.index.get_level_values('sid').unique()
    matched_data = []
    
    print(f"Testing with {len(urma_times)} URMA times and {len(stations)} stations")
    
    for i, urma_time in enumerate(urma_times[:5]):  # Test only first 5 times
        if i > 0 and i % 10 == 0:
            print(f"  Processed {i} URMA times...")
            
        window_start = urma_time - timedelta(hours=12)
        window_end = urma_time
        
        for station in stations:
            # Get URMA data
            try:
                urma_row = urma_subset.loc[(urma_time, station)]
                if isinstance(urma_row, pd.Series):
                    urma_data = urma_row.to_dict()
                else:
                    urma_data = urma_row.iloc[0].to_dict()
            except KeyError:
                continue
            
            # Find forecast matches
            forecast_mask = (
                (forecast_subset.index.get_level_values('valid_datetime') >= window_start) &
                (forecast_subset.index.get_level_values('valid_datetime') <= window_end) &
                (forecast_subset.index.get_level_values('sid') == station)
            )
            forecast_matches = forecast_subset[forecast_mask]
            
            # Find NBM matches
            nbm_mask = (
                (nbm_subset.index.get_level_values('valid_datetime') >= window_start) &
                (nbm_subset.index.get_level_values('valid_datetime') <= window_end) &
                (nbm_subset.index.get_level_values('sid') == station)
            )
            nbm_matches = nbm_subset[nbm_mask]
            
            # IMPROVED: Align by exact forecast time
            forecast_times = forecast_matches.index.get_level_values('valid_datetime').unique()
            
            for forecast_time in forecast_times:
                try:
                    forecast_row = forecast_matches.loc[(forecast_time, station)]
                    if isinstance(forecast_row, pd.DataFrame):
                        forecast_row = forecast_row.iloc[0]
                except KeyError:
                    continue
                
                try:
                    nbm_row = nbm_matches.loc[(forecast_time, station)]
                    if isinstance(nbm_row, pd.DataFrame):
                        nbm_row = nbm_row.iloc[0]
                except KeyError:
                    continue
                
                # Create properly aligned row
                combined_row = {
                    'urma_valid_datetime': urma_time,
                    'valid_datetime': forecast_time,
                    'sid': station,
                    'forecast_lead_hours': (forecast_time - urma_time).total_seconds() / 3600
                }
                combined_row.update(forecast_row.to_dict())
                combined_row.update(nbm_row.to_dict())
                combined_row.update(urma_data)
                matched_data.append(combined_row)
    
    if matched_data:
        result_df = pd.DataFrame(matched_data)
        result_df = result_df.set_index(['urma_valid_datetime', 'valid_datetime', 'sid'])
        return result_df
    else:
        return pd.DataFrame()

def test_alignment_verification(df, target_config):
    """Test the alignment verification logic."""
    print("Sample alignment verification:")
    
    gefs_col = 'gefs_' + target_config['forecast_col']
    nbm_col = 'nbm_tmax'  # Adjusted for test data
    urma_col = target_config['urma_obs_col']
    
    if all(col in df.columns for col in [gefs_col, nbm_col, urma_col]):
        print(f"GEFS-NBM correlation: {df[gefs_col].corr(df[nbm_col]):.3f}")
        
        if 'forecast_lead_hours' in df.columns:
            lead_hours = df['forecast_lead_hours'].unique()
            print(f"Lead hours found: {sorted(lead_hours)}")
    else:
        missing = [col for col in [gefs_col, nbm_col, urma_col] if col not in df.columns]
        print(f"Missing columns for verification: {missing}")

if __name__ == "__main__":
    quick_alignment_test()
