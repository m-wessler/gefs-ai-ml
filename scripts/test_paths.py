#!/usr/bin/env python3

"""Test script to verify file paths are working correctly."""

import pandas as pd
from pathlib import Path

# Configuration
BASE_PATH = 'data/'
STATION_IDS = ['KSLC', 'KBOI', 'KLAS', 'KSEA', 'KLAX']
FORECAST_HOURS = ['f024']

def test_file_paths():
    """Test if all expected files exist."""
    print(f"Testing file paths with BASE_PATH: {BASE_PATH}")
    print(f"Station IDs: {STATION_IDS}")
    print(f"Forecast hours: {FORECAST_HOURS}")
    print()
    
    forecast_found = 0
    nbm_found = 0
    
    # Test forecast and NBM data paths
    for data_type in ['forecast', 'nbm']:
        print(f"Checking {data_type} files:")
        for fhour in FORECAST_HOURS:
            data_dir = Path(BASE_PATH) / data_type / fhour
            for station_id in STATION_IDS:
                file_path = data_dir / f"{station_id}_2020_2025_{fhour}.csv"
                if file_path.exists():
                    print(f"  ✓ Found: {file_path}")
                    if data_type == 'forecast':
                        forecast_found += 1
                    else:
                        nbm_found += 1
                else:
                    print(f"  ✗ Missing: {file_path}")
        print()
    
    # Test URMA data path
    urma_path = Path(BASE_PATH) / 'urma' / 'WR_2020_2025.urma.parquet'
    if urma_path.exists():
        print(f"✓ Found URMA data: {urma_path}")
    else:
        print(f"✗ Missing URMA data: {urma_path}")
    
    print(f"\nSummary:")
    print(f"Forecast files found: {forecast_found}/{len(STATION_IDS)}")
    print(f"NBM files found: {nbm_found}/{len(STATION_IDS)}")
    print(f"URMA file found: {'Yes' if urma_path.exists() else 'No'}")

if __name__ == "__main__":
    test_file_paths()
