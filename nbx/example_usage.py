#!/usr/bin/env python3
"""
Example usage of the NBM Forecast Extractor
"""

import pandas as pd
from datetime import datetime
from nbm_forecast_extractor import NBMForecastExtractor

def main():
    """Example of how to use the NBM Forecast Extractor."""
    
    # Load your station metadata
    # Make sure your metadata.csv has columns: latitude, longitude, station_id
    try:
        metadata = pd.read_csv('./metadata.csv')
        print(f"Loaded metadata for {len(metadata)} stations")
    except FileNotFoundError:
        print("Please provide a metadata.csv file with station information")
        print("Required columns: latitude, longitude, station_id")
        return

    # Configuration
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 2)
    init_times = ['00', '06', '12', '18']  # t00z and t12z
    forecast_hours = ['024', '048', '072']  # 24, 48, and 72 hour forecasts

    print(f"\n📊 Configuration:")
    print(f"   Date range: {start_date.date()} to {end_date.date()}")
    print(f"   Init times: {init_times}")
    print(f"   Forecast hours: {forecast_hours}")
    print(f"   Stations: {len(metadata)}")

    # Initialize extractor
    extractor = NBMForecastExtractor(output_dir="./nbm_data")

    # Extract forecast data
    try:
        generated_files = extractor.extract_forecast_data_monthly(
            metadata_df=metadata,
            start_date=start_date,
            end_date=end_date,
            init_times=init_times,
            forecast_hours=forecast_hours,
            force_reprocess=False  # Set to True to reprocess existing files
        )
        
        print(f"\n✅ Extraction complete! Generated {len(generated_files)} files.")
        
        # List available files
        print("\n📁 Available files:")
        extractor.list_available_files()
        
        # Example: Load and examine the data
        if generated_files:
            print("\n📈 Loading first month of data for examination...")
            sample_data = extractor.load_monthly_data(year=2025, month=1)
            
            if not sample_data.empty:
                print(f"Sample data shape: {sample_data.shape}")
                print(f"Index levels: {sample_data.index.names}")
                print(f"Columns: {list(sample_data.columns)[:10]}...")  # Show first 10 columns
                print(f"Date range: {sample_data.index.get_level_values('forecast_time').min()} to {sample_data.index.get_level_values('forecast_time').max()}")
                
                # Show sample of the data
                print("\nSample data (first 5 rows):")
                print(sample_data.head())
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        raise

if __name__ == "__main__":
    main()