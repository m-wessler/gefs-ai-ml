#!/usr/bin/env python3

"""
Improved data alignment functions for GEFS ML trainer that properly handle
aggregation across forecast hours.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

def create_time_matched_dataset_improved(forecast_subset, nbm_subset, urma_subset):
    """
    Improved time-matched dataset creation that properly handles aggregation across forecast hours.
    
    Key improvements:
    1. Aggregates multiple forecast hours for the same verification time
    2. Ensures proper NBM-GEFS alignment by forecast hour
    3. Creates separate entries for each forecast hour to maintain independence
    """
    urma_times = urma_subset.index.get_level_values('valid_datetime').unique()
    stations = urma_subset.index.get_level_values('sid').unique()
    matched_data = []

    print("Creating improved time-matched dataset...")
    print(f"Processing {len(urma_times)} URMA times for {len(stations)} stations")
    
    # Track alignment statistics
    total_matches = 0
    total_urma_obs = 0
    forecast_hour_counts = {}
    
    for urma_time in urma_times:
        total_urma_obs += len(stations)
        window_start = urma_time - timedelta(hours=12)
        window_end = urma_time

        for station in stations:
            # Get URMA data for this time/station
            try:
                urma_row = urma_subset.loc[(urma_time, station)]
                if isinstance(urma_row, pd.Series):
                    urma_data = urma_row.to_dict()
                else:
                    urma_data = urma_row.iloc[0].to_dict()
            except KeyError:
                continue

            # Find matching forecast data within window
            forecast_mask = (
                (forecast_subset.index.get_level_values('valid_datetime') >= window_start) &
                (forecast_subset.index.get_level_values('valid_datetime') <= window_end) &
                (forecast_subset.index.get_level_values('sid') == station)
            )
            forecast_matches = forecast_subset[forecast_mask]

            # Find matching NBM data within window
            nbm_mask = (
                (nbm_subset.index.get_level_values('valid_datetime') >= window_start) &
                (nbm_subset.index.get_level_values('valid_datetime') <= window_end) &
                (nbm_subset.index.get_level_values('sid') == station)
            )
            nbm_matches = nbm_subset[nbm_mask]

            # IMPROVEMENT: Group by forecast valid time to ensure proper alignment
            forecast_times = forecast_matches.index.get_level_values('valid_datetime').unique()
            
            for forecast_time in forecast_times:
                # Get forecast data for this specific time
                try:
                    forecast_row = forecast_matches.loc[(forecast_time, station)]
                    if isinstance(forecast_row, pd.DataFrame):
                        forecast_row = forecast_row.iloc[0]
                except KeyError:
                    continue
                
                # Find corresponding NBM data for the same forecast time
                try:
                    nbm_row = nbm_matches.loc[(forecast_time, station)]
                    if isinstance(nbm_row, pd.DataFrame):
                        nbm_row = nbm_row.iloc[0]
                except KeyError:
                    # If no exact NBM match, skip this forecast
                    continue
                
                # Create combined row with proper alignment
                combined_row = {
                    'urma_valid_datetime': urma_time,
                    'valid_datetime': forecast_time,
                    'sid': station,
                    'forecast_lead_hours': (forecast_time - urma_time).total_seconds() / 3600  # Add lead time info
                }
                combined_row.update(forecast_row.to_dict())
                combined_row.update(nbm_row.to_dict())
                combined_row.update(urma_data)
                matched_data.append(combined_row)
                total_matches += 1
                
                # Track forecast hour distribution
                lead_hours = combined_row['forecast_lead_hours']
                forecast_hour_counts[lead_hours] = forecast_hour_counts.get(lead_hours, 0) + 1

    if matched_data:
        result_df = pd.DataFrame(matched_data)
        result_df = result_df.set_index(['urma_valid_datetime', 'valid_datetime', 'sid'])
        
        print(f"Created improved time-matched dataset:")
        print(f"  Total records: {len(result_df)}")
        print(f"  Match rate: {total_matches/total_urma_obs:.2%}")
        print(f"  Forecast hour distribution: {forecast_hour_counts}")
        
        return result_df
    else:
        print("Warning: No time-matched data created")
        return pd.DataFrame()

def verify_data_alignment_improved(df, target_config):
    """Enhanced verification that checks forecast hour alignment specifically."""
    print("\n=== IMPROVED DATA ALIGNMENT VERIFICATION ===")
    
    # Get the key columns
    gefs_col = 'gefs_' + target_config['forecast_col']
    nbm_col = 'nbm_' + target_config['forecast_col'] 
    urma_col = target_config['urma_obs_col']
    
    # Check if columns exist
    missing_cols = []
    for col in [gefs_col, nbm_col, urma_col]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        return
    
    # Enhanced sample verification with forecast lead time info
    print("Sample alignment check with forecast lead times:")
    print("Format: URMA_time | Forecast_time | Lead_hrs | Station | GEFS | NBM | URMA")
    
    sample_df = df.head(10).copy()
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        if i >= 5:
            break
        urma_time, valid_time, station = idx
        lead_hours = row.get('forecast_lead_hours', 'N/A')
        gefs_val = row.get(gefs_col, 'N/A')
        nbm_val = row.get(nbm_col, 'N/A')
        urma_val = row.get(urma_col, 'N/A')
        
        print(f"{i+1:2d}: {urma_time.strftime('%Y-%m-%d %H:%M')} | "
              f"{valid_time.strftime('%Y-%m-%d %H:%M')} | "
              f"{lead_hours:6.1f} | {station} | "
              f"GEFS: {gefs_val:6.1f} | NBM: {nbm_val:6.1f} | URMA: {urma_val:6.1f}")
    
    # Check forecast hour distribution
    if 'forecast_lead_hours' in df.columns:
        lead_hour_counts = df['forecast_lead_hours'].value_counts().sort_index()
        print(f"\nForecast lead hour distribution:")
        for lead_hour, count in lead_hour_counts.items():
            print(f"  {lead_hour:6.1f} hours: {count:6d} records")
    
    # Check for alignment by forecast hour
    print("\nAlignment verification by forecast hour:")
    if 'forecast_lead_hours' in df.columns:
        for lead_hour in df['forecast_lead_hours'].unique():
            subset = df[df['forecast_lead_hours'] == lead_hour]
            if len(subset) > 10:  # Only check if sufficient data
                correlation = subset[gefs_col].corr(subset[nbm_col])
                print(f"  {lead_hour:6.1f}h: GEFS-NBM correlation = {correlation:.3f} ({len(subset)} samples)")
                
                if correlation < 0.8:
                    print(f"    WARNING: Low correlation at {lead_hour}h suggests alignment issues!")
    
    # Overall data quality checks
    print("\nOverall Data Quality Checks:")
    
    # Check for missing data patterns
    gefs_missing = df[gefs_col].isna().sum()
    nbm_missing = df[nbm_col].isna().sum()
    urma_missing = df[urma_col].isna().sum()
    
    print(f"Missing data - GEFS: {gefs_missing}, NBM: {nbm_missing}, URMA: {urma_missing}")
    
    # Check for duplicate forecast times per URMA time/station
    duplicates = df.reset_index().groupby(['urma_valid_datetime', 'sid']).size()
    max_forecasts_per_obs = duplicates.max()
    avg_forecasts_per_obs = duplicates.mean()
    
    print(f"Forecasts per observation - Max: {max_forecasts_per_obs}, Avg: {avg_forecasts_per_obs:.1f}")
    
    # Overall correlation check
    if not df[gefs_col].isna().all() and not df[nbm_col].isna().all():
        overall_correlation = df[gefs_col].corr(df[nbm_col])
        print(f"Overall GEFS-NBM correlation: {overall_correlation:.3f}")
        
        if overall_correlation < 0.8:
            print("WARNING: Overall low GEFS-NBM correlation suggests systematic alignment issues!")
    
    print("=== END IMPROVED VERIFICATION ===\n")

def aggregate_forecast_hours_for_evaluation(df, method='ensemble'):
    """
    Aggregate multiple forecast hours for the same URMA observation.
    
    Parameters:
    - df: DataFrame with multiple forecast hours per URMA time
    - method: 'ensemble' (average), 'best_lead' (closest to verification), or 'separate' (keep separate)
    """
    if method == 'separate':
        # Keep forecast hours separate - this maintains more training data
        return df
    
    elif method == 'ensemble':
        # Average multiple forecast hours for the same URMA observation
        print("Aggregating forecast hours using ensemble averaging...")
        
        # Group by URMA time and station, average the forecasts
        aggregated = df.groupby(['urma_valid_datetime', 'sid']).agg({
            col: 'mean' for col in df.columns if col not in ['urma_valid_datetime', 'sid']
        }).reset_index()
        
        # Set new index
        aggregated = aggregated.set_index(['urma_valid_datetime', 'sid'])
        print(f"Aggregated from {len(df)} to {len(aggregated)} records")
        return aggregated
    
    elif method == 'best_lead':
        # Keep only the forecast hour closest to 24 hours (optimal lead time)
        print("Selecting best lead time forecasts...")
        
        if 'forecast_lead_hours' in df.columns:
            # Find forecast closest to 24 hours for each URMA observation
            def select_best_lead(group):
                target_lead = 24.0  # Target 24-hour lead time
                best_idx = (group['forecast_lead_hours'] - target_lead).abs().idxmin()
                return group.loc[best_idx]
            
            best_forecasts = df.groupby(['urma_valid_datetime', 'sid']).apply(select_best_lead)
            print(f"Selected best lead times from {len(df)} to {len(best_forecasts)} records")
            return best_forecasts
        else:
            print("Warning: No forecast_lead_hours column found, returning original data")
            return df
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
