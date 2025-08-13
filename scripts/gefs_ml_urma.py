#!/home/chad.kahler/anaconda3/envs/dev/bin/python

"""
GEFS ML Weather Forecasting Post-Processing Pipeline

This script implements a machine learning approach to post-process weather forecasts,
specifically focusing on temperature predictions (maximum or minimum). It combines GEFS (Global Ensemble
Forecast System), NBM (National Blend of Models), and URMA (Unrestricted Mesoscale Analysis)
data to train ML models that can improve forecast accuracy.

Target Variable Configuration:
- Set TARGET_VARIABLE = 'tmax' for maximum temperature predictions
- Set TARGET_VARIABLE = 'tmin' for minimum temperature predictions

Main workflow:
1. Load and combine forecast, NBM, and URMA data
2. Create time-matched datasets with quality control
3. Prepare data for ML training
4. Train and evaluate Random Forest model
5. Compare performance against baseline NBM forecasts

Author: Generated from Jupyter notebook
Date: 2025
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================

# Base directory for all data files
BASE_PATH = '/nas/stid/data/gefs-ml/'

def discover_available_stations(base_path=None, forecast_hours=None, data_types=None):
    """
    Automatically discover available station IDs by scanning data directory structure.
    
    This function searches through the forecast and NBM data directories to find
    all available station files and extract unique station identifiers.
    
    Parameters:
    -----------
    base_path : str or None
        Base directory path where data folders are located. If None, uses BASE_PATH.
    forecast_hours : List[str] or None
        List of forecast hours to check (e.g., ['f024', 'f048']). If None, discovers all.
    data_types : List[str] or None
        Data types to check ('forecast', 'nbm'). If None, checks both.
        
    Returns:
    --------
    dict: Contains discovered stations, forecast hours, and statistics
    """
    if base_path is None:
        base_path = BASE_PATH
    if data_types is None:
        data_types = ['forecast', 'nbm']
    
    base_path = Path(base_path)
    discovered_info = {
        'stations': set(),
        'forecast_hours': set(),
        'files_found': 0,
        'data_types_found': {},
        'station_coverage': {}
    }
    
    print(f"Scanning for available stations in: {base_path}")
    
    # Check each data type directory
    for data_type in data_types:
        data_dir = base_path / data_type
        if not data_dir.exists():
            print(f"Warning: {data_type} directory not found: {data_dir}")
            continue
            
        discovered_info['data_types_found'][data_type] = {
            'forecast_hours': set(),
            'stations': set()
        }
        
        # Get forecast hour directories
        fhour_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('f')]
        
        if forecast_hours is not None:
            # Filter to only requested forecast hours
            fhour_dirs = [d for d in fhour_dirs if d.name in forecast_hours]
        
        # Scan each forecast hour directory
        for fhour_dir in sorted(fhour_dirs):
            fhour = fhour_dir.name
            discovered_info['forecast_hours'].add(fhour)
            discovered_info['data_types_found'][data_type]['forecast_hours'].add(fhour)
            
            # Find CSV files in this directory
            csv_files = list(fhour_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                discovered_info['files_found'] += 1
                
                # Extract station ID from filename pattern: STATION_2020_2025_fXXX.csv
                filename = csv_file.name
                if '_2020_2025_' in filename and filename.endswith('.csv'):
                    station_id = filename.split('_2020_2025_')[0]
                    discovered_info['stations'].add(station_id)
                    discovered_info['data_types_found'][data_type]['stations'].add(station_id)
                    
                    # Track station coverage across forecast hours
                    if station_id not in discovered_info['station_coverage']:
                        discovered_info['station_coverage'][station_id] = set()
                    discovered_info['station_coverage'][station_id].add(fhour)
    
    # Convert sets to sorted lists for easier use
    discovered_info['stations'] = sorted(list(discovered_info['stations']))
    discovered_info['forecast_hours'] = sorted(list(discovered_info['forecast_hours']))
    
    # Print discovery summary
    print(f"\nDiscovery Results:")
    print(f"  Total stations found: {len(discovered_info['stations'])}")
    print(f"  Total forecast hours found: {len(discovered_info['forecast_hours'])}")
    print(f"  Total files scanned: {discovered_info['files_found']}")
    
    if discovered_info['stations']:
        print(f"\nAvailable stations: {discovered_info['stations']}")
        print(f"Available forecast hours: {discovered_info['forecast_hours']}")
        
        # Show station coverage statistics
        complete_stations = []
        partial_stations = []
        total_fhours = len(discovered_info['forecast_hours'])
        
        for station, fhours in discovered_info['station_coverage'].items():
            if len(fhours) == total_fhours:
                complete_stations.append(station)
            else:
                partial_stations.append(f"{station}({len(fhours)}/{total_fhours})")
        
        if complete_stations:
            print(f"\nStations with complete forecast hour coverage: {complete_stations}")
        if partial_stations:
            print(f"Stations with partial coverage: {partial_stations}")
    
    return discovered_info

def list_available_stations(base_path=None, forecast_hours=None, min_coverage=1.0):
    """
    Convenience function to list available stations with filtering options.
    
    Parameters:
    -----------
    base_path : str or None
        Base directory path where data folders are located. If None, uses BASE_PATH.
    forecast_hours : List[str] or None
        Specific forecast hours to check. If None, checks all available.
    min_coverage : float
        Minimum fraction of forecast hours a station must have (0.0 to 1.0).
        1.0 = station must have data for ALL forecast hours
        0.5 = station must have data for at least 50% of forecast hours
        
    Returns:
    --------
    List[str]: Station IDs that meet the coverage criteria
    """
    discovery = discover_available_stations(base_path, forecast_hours)
    
    if not discovery['stations']:
        return []
    
    total_fhours = len(discovery['forecast_hours'])
    required_fhours = int(total_fhours * min_coverage)
    
    qualified_stations = []
    for station, fhours in discovery['station_coverage'].items():
        if len(fhours) >= required_fhours:
            qualified_stations.append(station)
    
    print(f"\nFiltered Results (min coverage {min_coverage*100:.0f}%):")
    print(f"  Qualified stations: {qualified_stations}")
    print(f"  ({len(qualified_stations)}/{len(discovery['stations'])} stations qualify)")
    
    return qualified_stations

# Automatically discover available stations (set AUTO_DISCOVER_STATIONS = False to use manual list)
AUTO_DISCOVER_STATIONS = True

if AUTO_DISCOVER_STATIONS:
    try:
        station_discovery = discover_available_stations()
        STATION_IDS = station_discovery['stations']
        if not STATION_IDS:
            print("Warning: No stations discovered automatically, using manual list")
    except Exception as e:
        print(f"Error during station discovery: {e}")
else:
    # Manual station list - MODIFY THIS LIST AS NEEDED
    STATION_IDS = ['KSLC', 'KBOI', 'KEKO', 'KSEA', 'KLAX']

# Forecast hours to include - MODIFY THIS LIST AS NEEDED
FORECAST_HOURS = ['f024']

# Quality control threshold (maximum allowed deviation between URMA and station obs in °C)
QC_THRESHOLD = 5.0

# Target variable type - MODIFY THIS TO CHANGE TARGET VARIABLE
# Options: 'tmax' for maximum temperature, 'tmin' for minimum temperature
TARGET_VARIABLE = 'tmax'

# Post-processing mode toggle - MODIFY THIS TO CHANGE APPROACH
# True: Use NBM forecasts as predictors (post-processing approach)
# False: Exclude NBM forecasts, use only meteorological features (independent forecasting)
POSTPROCESS_NBM = False

# =============================================================================
# TARGET VARIABLE CONFIGURATION FUNCTIONS
# =============================================================================

def get_target_config(target_type=None):
    """
    Get configuration for target variable based on type (tmax or tmin).
    
    Parameters:
    -----------
    target_type : str or None
        Target variable type ('tmax' or 'tmin'). If None, uses TARGET_VARIABLE.
        
    Returns:
    --------
    dict: Configuration with column names and URMA field mapping
    """
    if target_type is None:
        target_type = TARGET_VARIABLE
    if target_type.lower() in ['tmax', 'maxt']:
        return {
            'target_type': 'tmax',
            'forecast_col': 'tmax_2m',
            'obs_col': 'tmax_obs', 
            'gefs_obs_col': 'gefs_tmax_obs',
            'nbm_col': 'nbm_tmax_2m',
            'nbm_obs_col': 'nbm_tmax_obs',
            'urma_obs_col': 'urma_tmax_obs',
            'urma_field': 'Maximum temperature_2_heightAboveGround',
            'description': 'maximum temperature'
        }
    elif target_type.lower() in ['tmin', 'mint']:
        return {
            'target_type': 'tmin',
            'forecast_col': 'tmin_2m',
            'obs_col': 'tmin_obs',
            'gefs_obs_col': 'gefs_tmin_obs', 
            'nbm_col': 'nbm_tmin_2m',
            'nbm_obs_col': 'nbm_tmin_obs',
            'urma_obs_col': 'urma_tmin_obs',
            'urma_field': 'Minimum temperature_2_heightAboveGround',
            'description': 'minimum temperature'
        }
    else:
        raise ValueError(f"Target type '{target_type}' not supported. Use 'tmax' or 'tmin'.")

# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================

def load_combined_data(
    station_ids: List[str] = None,
    forecast_hours: List[str] = None,
    base_path: str = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and combine forecast, NBM, and URMA data into three DataFrames.
    
    This function reads CSV files for forecast and NBM data organized by forecast hour
    and station ID, and reads the URMA parquet file. All data is indexed by 
    [valid_datetime, sid/station_id] for consistent merging.

    Parameters:
    -----------
    station_ids : List[str] or None
        List of station IDs to load (e.g., ['KSLC', 'KBOI']). If None, uses STATION_IDS.
    forecast_hours : List[str] or None
        List of forecast hours to load (e.g., ['f120']). If None, uses FORECAST_HOURS.
    base_path : str or None
        Base directory path where data folders are located. If None, uses BASE_PATH.

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        forecast_df, nbm_df, urma_df with MultiIndex [valid_datetime, sid/station_id]
    """
    # Use defaults if not provided
    if station_ids is None:
        station_ids = STATION_IDS
    if forecast_hours is None:
        forecast_hours = FORECAST_HOURS
    if base_path is None:
        base_path = BASE_PATH
        
    print(f"Loading data from: {base_path}")
    print(f"Station IDs: {station_ids}")
    print(f"Forecast hours: {forecast_hours}")
    
    forecast_dfs = []
    nbm_dfs = []

    # Load forecast and NBM data from CSV files
    for data_type in ['forecast', 'nbm']:
        for fhour in forecast_hours:
            for station in station_ids:
                # Construct file path: base_path/forecast/f120/KSLC_2020_2025_f120.csv
                filename = f"{station}_2020_2025_{fhour}.csv"
                filepath = Path(base_path) / data_type / fhour / filename

                try:
                    df = pd.read_csv(filepath)
                    df['sid'] = station  # Add station ID column
                    
                    if data_type == 'forecast':
                        forecast_dfs.append(df)
                    else:
                        nbm_dfs.append(df)
                        
                except FileNotFoundError:
                    print(f"Warning: File not found: {filepath}")

    # Combine all forecast and NBM DataFrames
    forecast_df = pd.concat(forecast_dfs, ignore_index=True)
    nbm_df = pd.concat(nbm_dfs, ignore_index=True)

    # Convert datetime columns and set MultiIndex for consistent merging
    forecast_df['valid_datetime'] = pd.to_datetime(forecast_df['valid_datetime'])
    nbm_df['valid_datetime'] = pd.to_datetime(nbm_df['valid_datetime'])
    
    forecast_df = forecast_df.set_index(['valid_datetime', 'sid'])
    nbm_df = nbm_df.set_index(['valid_datetime', 'sid'])

    # Load URMA data (already processed and in parquet format)
    urma_path = Path(base_path) / 'urma' / 'WR_2020_2025.urma.parquet'
    print(f"Loading URMA data from: {urma_path}")
    urma_df = pd.read_parquet(urma_path, engine='pyarrow')

    # Filter URMA for our selected stations only
    urma_df = urma_df.loc[urma_df.index.get_level_values(1).isin(station_ids)]
    
    print(f"Loaded forecast data: {forecast_df.shape}")
    print(f"Loaded NBM data: {nbm_df.shape}")
    print(f"Loaded URMA data: {urma_df.shape}")

    return forecast_df, nbm_df, urma_df

def create_time_matched_dataset(forecast_subset, nbm_subset, urma_subset):
    """
    Create a dataset with MultiIndex [urma_valid_datetime, valid_datetime, sid]
    where forecast/NBM valid_datetime entries are within 12 hours prior to urma_valid_datetime.
    
    This function addresses the temporal mismatch between forecasts (issued at specific times)
    and URMA analysis (valid at specific times) by creating a window-based matching approach.
    
    For each URMA analysis time, we look for all forecasts issued in the 12 hours prior
    to that analysis time. This allows multiple forecasts to be matched to each URMA observation,
    providing more training data and capturing forecast evolution.

    Parameters:
    -----------
    forecast_subset : pd.DataFrame
        GEFS forecast data with MultiIndex [valid_datetime, sid]
    nbm_subset : pd.DataFrame  
        NBM forecast data with MultiIndex [valid_datetime, sid]
    urma_subset : pd.DataFrame
        URMA analysis data with MultiIndex [valid_datetime, sid]

    Returns:
    --------
    pd.DataFrame
        Time-matched dataset with MultiIndex [urma_valid_datetime, valid_datetime, sid]
        where forecast data is merged with corresponding URMA analysis
    """
    # Get unique URMA times and stations for iteration
    urma_times = urma_subset.index.get_level_values('valid_datetime').unique()
    stations = urma_subset.index.get_level_values('sid').unique()

    matched_data = []

    print("Creating time-matched dataset...")
    for urma_time in urma_times:
        # Define 12-hour window prior to URMA analysis time
        window_start = urma_time - timedelta(hours=12)
        window_end = urma_time

        for station in stations:
            # Get URMA analysis data for this time/station combination
            try:
                urma_data = urma_subset.loc[(urma_time, station)]
            except KeyError:
                continue  # Skip if no URMA data for this time/station

            # Get forecast and NBM data within the time window for this station
            forecast_windowed = pd.DataFrame()
            nbm_windowed = pd.DataFrame()

            # Extract forecast data within time window
            try:
                forecast_station_data = forecast_subset.xs(station, level='sid')
                mask = (forecast_station_data.index >= window_start) & (forecast_station_data.index <= window_end)
                forecast_windowed = forecast_station_data[mask].copy()
            except KeyError:
                pass

            # Extract NBM data within time window  
            try:
                nbm_station_data = nbm_subset.xs(station, level='sid')
                mask = (nbm_station_data.index >= window_start) & (nbm_station_data.index <= window_end)
                nbm_windowed = nbm_station_data[mask].copy()
            except KeyError:
                pass

            # Get all unique forecast valid times within the window
            all_times = set()
            if not forecast_windowed.empty:
                all_times.update(forecast_windowed.index)
            if not nbm_windowed.empty:
                all_times.update(nbm_windowed.index)

            # For each valid time, merge forecast and NBM data
            for valid_time in all_times:
                matched_row = {
                    'urma_valid_datetime': urma_time,
                    'valid_datetime': valid_time,
                    'sid': station
                }

                # Combine forecast and NBM data, taking max for overlapping columns
                combined_data = {}

                # Add forecast data if available for this valid time
                if not forecast_windowed.empty and valid_time in forecast_windowed.index:
                    for col, val in forecast_windowed.loc[valid_time].items():
                        combined_data[col] = val

                # Add NBM data if available, taking max for overlapping columns
                if not nbm_windowed.empty and valid_time in nbm_windowed.index:
                    for col, val in nbm_windowed.loc[valid_time].items():
                        if col in combined_data:
                            # Take maximum value for overlapping columns (conservative approach)
                            combined_data[col] = max(combined_data[col], val)
                        else:
                            combined_data[col] = val

                # Add the combined forecast/NBM data to the row
                matched_row.update(combined_data)

                # Add URMA analysis data
                for col, val in urma_data.items():
                    matched_row[col] = val

                matched_data.append(matched_row)

    # Create DataFrame with new MultiIndex structure
    if matched_data:
        result_df = pd.DataFrame(matched_data)
        result_df = result_df.set_index(['urma_valid_datetime', 'valid_datetime', 'sid'])
        print(f"Created time-matched dataset with {len(result_df)} records")
        return result_df
    else:
        print("Warning: No time-matched data created")
        return pd.DataFrame()

def apply_qc_filter(df, urma_obs_threshold=QC_THRESHOLD, target_config=None):
    """
    Apply quality control filter based on deviation between URMA and station observations.
    
    This filter removes data points where URMA analysis and station observations disagree
    significantly, which often indicates data quality issues or spatial representativeness problems.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with temperature data
    urma_obs_threshold : float
        Maximum allowed deviation between URMA and observations (default from config)
    target_config : dict or None
        Target configuration dict from get_target_config(). If None, uses current TARGET_VARIABLE.

    Returns:
    --------
    pd.DataFrame : QC filtered dataframe
    dict : QC statistics and diagnostics
    """
    if target_config is None:
        target_config = get_target_config()
        
    obs_col = target_config['gefs_obs_col']
    urma_col = target_config['urma_obs_col']
    df_work = df.copy()
    df_work['urma_obs_deviation'] = abs(df_work[urma_col] - df_work[obs_col])

    # Identify records that pass QC threshold
    qc_mask = df_work['urma_obs_deviation'] <= urma_obs_threshold

    # Apply QC filter
    df_qc = df_work[qc_mask].copy()
    df_qc = df_qc.drop('urma_obs_deviation', axis=1)  # Remove temporary column

    # Calculate QC statistics for reporting
    total_records = len(df_work)
    records_with_both = df_work[[obs_col, urma_col]].dropna().shape[0]
    qc_passed = qc_mask.sum()
    qc_failed = (~qc_mask).sum()

    qc_stats = {
        'total_records': total_records,
        'records_with_both_obs_urma': records_with_both,
        'qc_passed': qc_passed,
        'qc_failed': qc_failed,
        'qc_pass_rate': qc_passed / records_with_both if records_with_both > 0 else 0,
        'threshold_used': urma_obs_threshold,
        'max_deviation_in_passed': df_work[qc_mask]['urma_obs_deviation'].max() if qc_passed > 0 else None,
        'mean_deviation_in_failed': df_work[~qc_mask]['urma_obs_deviation'].mean() if qc_failed > 0 else None
    }

    return df_qc, qc_stats

# =============================================================================
# MACHINE LEARNING PIPELINE FUNCTIONS
# =============================================================================

def identify_and_drop_non_predictive_columns(df):
    """
    Identify and remove columns that are not suitable for ML prediction.
    
    Removes:
    - Non-numeric columns
    - Columns with only one unique value (constants)
    - Station metadata columns (lat, lon, elevation)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with potential features

    Returns:
    --------
    pd.DataFrame : Cleaned dataframe with only predictive features
    dict : Information about dropped columns and reasons
    """
    original_cols = df.columns.tolist()
    cols_to_drop = []
    drop_reasons = {}

    # Identify potential station metadata columns
    metadata_keywords = ['lat', 'lon', 'elevation', 'elev', 'alt', 'height']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in metadata_keywords):
            cols_to_drop.append(col)
            drop_reasons[col] = "Station metadata"

    for col in df.columns:
        if col in cols_to_drop:
            continue
            
        # Check if column is non-numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            cols_to_drop.append(col)
            drop_reasons[col] = f"Non-numeric dtype ({df[col].dtype})"
            continue

        # Check if column has only one unique value (constant)
        unique_vals = df[col].nunique()
        if unique_vals <= 1:
            cols_to_drop.append(col)
            drop_reasons[col] = f"Globally constant (nunique={unique_vals})"
            continue

    # Drop identified columns
    cleaned_df = df.drop(columns=cols_to_drop)

    drop_info = {
        'original_shape': df.shape,
        'cleaned_shape': cleaned_df.shape,
        'dropped_columns': cols_to_drop,
        'drop_reasons': drop_reasons,
        'kept_columns': cleaned_df.columns.tolist(),
    }

    return cleaned_df, drop_info

def combine_features_and_targets(feature_data, target_data, target_config=None):
    """
    Combine feature data and target data along matching indices for ML training.
    
    This function merges forecast features with target observations, removing
    observation-based features that won't be available during real-time prediction.

    Parameters:
    -----------
    feature_data : pd.DataFrame
        Features with MultiIndex [valid_datetime, sid]
    target_data : pd.DataFrame
        Target data with MultiIndex [valid_datetime, sid]
    target_config : dict or None
        Target configuration dict from get_target_config(). If None, uses current TARGET_VARIABLE.

    Returns:
    --------
    pd.DataFrame : Combined dataset ready for ML
    dict : Merge statistics and information
    """
    if target_config is None:
        target_config = get_target_config()
        
    # Use the GEFS observations as the primary target, URMA for reference
    target_cols = [target_config['gefs_obs_col']]  # gefs_tmax_obs
    print("=== COMBINING FEATURES AND TARGETS ===")
    print(f"Feature data shape: {feature_data.shape}")
    print(f"Target data shape: {target_data.shape}")

    # Handle target column specification
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    # Validate target columns exist
    target_keywords = [target_config['target_type'], 'temp', 'target']
    available_targets = [col for col in target_data.columns if any(keyword in col.lower() for keyword in target_keywords)]
    missing_targets = [col for col in target_cols if col not in target_data.columns]

    if missing_targets:
        print(f"WARNING: Target columns not found: {missing_targets}")
        print(f"Available target-like columns: {available_targets}")
        target_cols = [col for col in target_cols if col in target_data.columns]
        if not target_cols:
            raise ValueError("No valid target columns found")

    print(f"Using target columns: {target_cols}")

    # Extract target columns and find common indices
    target_subset = target_data[target_cols]
    common_indices = feature_data.index.intersection(target_subset.index)

    # Align both datasets to common indices
    features_aligned = feature_data.loc[common_indices]
    targets_aligned = target_subset.loc[common_indices]

    # Create combined dataset
    combined_data = features_aligned.copy()

    # Remove observation-based features (not available for real-time prediction)
    obs_features = [col for col in combined_data.columns if 'obs' in col.lower()]
    if obs_features:
        print(f"Removing {len(obs_features)} observation-based features: {obs_features}")
        combined_data = combined_data.drop(columns=obs_features)

    # Add target columns
    for target_col in target_cols:
        combined_data[target_col] = targets_aligned[target_col]

    # Remove rows with NaN in ANY target column
    initial_rows = len(combined_data)
    combined_data = combined_data.dropna(subset=target_cols)
    final_rows = len(combined_data)

    merge_stats = {
        'feature_data_rows': len(feature_data),
        'target_data_rows': len(target_data),
        'common_indices': len(common_indices),
        'initial_combined_rows': initial_rows,
        'final_combined_rows': final_rows,
        'rows_dropped_nan_targets': initial_rows - final_rows,
        'removed_obs_features': obs_features,
        'target_columns_used': target_cols,
        'feature_columns': len(features_aligned.columns) - len(obs_features),
        'final_columns': len(combined_data.columns)
    }

    return combined_data, merge_stats

def prepare_postprocessing_data(ml_dataset, target_config=None):
    """
    Prepare data for ML training by separating features, forecasts, and targets.
    
    Depending on POSTPROCESS_NBM setting:
    - True: Use NBM forecasts as predictors (post-processing approach) 
    - False: Exclude NBM forecasts, use only meteorological features (independent forecasting)

    Parameters:
    -----------
    ml_dataset : pd.DataFrame
        Combined dataset with features and targets
    target_config : dict or None
        Target configuration dict from get_target_config(). If None, uses current TARGET_VARIABLE.

    Returns:
    --------
    dict: Contains separated datasets and baseline performance metrics
    """
    if target_config is None:
        target_config = get_target_config()
        
    # Use the actual column names that exist in the dataset (with prefixes)
    target_col = target_config['gefs_obs_col']  # gefs_tmax_obs (the actual target)
    raw_forecast_col = 'gefs_' + target_config['forecast_col']  # gefs_tmax_2m
    nbm_forecast_col = 'nbm_' + target_config['forecast_col']  # nbm_tmax_2m
    
    df = ml_dataset.copy()
    df = df.dropna(subset=[target_col])  # Remove rows with missing targets

    # Calculate baseline metrics for existing forecast models
    baseline_metrics = {}
    forecast_cols = []

    print(f"ML Mode: {'Post-processing NBM' if POSTPROCESS_NBM else 'Independent forecasting (no NBM)'}")

    # Always evaluate NBM forecast for baseline comparison if available
    if nbm_forecast_col is not None and nbm_forecast_col in df.columns:
        df['nbm_bias'] = df[target_col] - df[nbm_forecast_col]
        df['nbm_abs_error'] = np.abs(df['nbm_bias'])

        nbm_mae = df['nbm_abs_error'].mean()
        nbm_rmse = np.sqrt((df['nbm_bias']**2).mean())
        baseline_metrics['nbm_mae'] = nbm_mae
        baseline_metrics['nbm_rmse'] = nbm_rmse
        
        # Only include NBM as a feature if post-processing is enabled
        if POSTPROCESS_NBM:
            forecast_cols.append(nbm_forecast_col)

        print("Baseline Performance:")
        print(f"NBM Model - MAE: {nbm_mae:.3f}°C, RMSE: {nbm_rmse:.3f}°C")
        baseline_printed = True
    else:
        baseline_printed = False

    # Evaluate raw forecast if available
    if raw_forecast_col is not None and raw_forecast_col in df.columns:
        df['raw_bias'] = df[target_col] - df[raw_forecast_col]
        df['raw_abs_error'] = np.abs(df['raw_bias'])

        raw_mae = df['raw_abs_error'].mean()
        raw_rmse = np.sqrt((df['raw_bias']**2).mean())
        baseline_metrics['raw_mae'] = raw_mae
        baseline_metrics['raw_rmse'] = raw_rmse
        forecast_cols.append(raw_forecast_col)

        if not baseline_printed:  # Only print header if NBM wasn't already shown
            print("Baseline Performance:")
        print(f"Raw Model - MAE: {raw_mae:.3f}°C, RMSE: {raw_rmse:.3f}°C")

    # Separate features, forecasts, and targets
    exclude_cols = [target_col] + forecast_cols + [
        'nbm_bias', 'nbm_abs_error', 'raw_bias', 'raw_abs_error'
    ]
    
    # If not post-processing, also exclude NBM forecast columns from features
    # but keep them in the dataset for baseline evaluation
    if not POSTPROCESS_NBM:
        nbm_cols_in_data = [col for col in df.columns if 'nbm_' in col and col not in exclude_cols]
        exclude_cols.extend(nbm_cols_in_data)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X_features = df[feature_cols].copy()

    # Create forecast features matrix (only includes forecast columns used as features)
    if forecast_cols:
        X_forecasts = df[forecast_cols].copy()
        X_combined = pd.concat([X_forecasts, X_features], axis=1)
    else:
        X_forecasts = pd.DataFrame(index=df.index)
        X_combined = X_features.copy()

    y = df[target_col].copy()

    return {
        'X_combined': X_combined,
        'X_features': X_features,
        'X_forecasts': X_forecasts,
        'y': y,
        'df_full': df,  # This contains all columns including NBM for evaluation
        'feature_cols': feature_cols,
        'forecast_cols': forecast_cols,
        'baseline_metrics': baseline_metrics
    }

def create_time_based_splits(data_dict, test_size=0.2, val_size=0.1):
    """
    Create time-based train/validation/test splits to avoid data leakage.
    
    Time-based splitting ensures that the model is trained on past data and
    evaluated on future data, which is how it would be used in practice.

    Parameters:
    -----------
    data_dict : dict
        Data dictionary from prepare_postprocessing_data
    test_size : float
        Fraction of data for testing (most recent data)
    val_size : float
        Fraction of remaining data for validation

    Returns:
    --------
    dict: Train/validation/test splits with date ranges
    """
    df = data_dict['df_full']

    # Get unique dates and sort chronologically
    unique_dates = df.index.get_level_values('valid_datetime').unique().sort_values()
    n_dates = len(unique_dates)

    # Calculate split points (most recent data for testing)
    test_start_idx = int(n_dates * (1 - test_size))
    val_start_idx = int((n_dates - int(n_dates * test_size)) * (1 - val_size))

    train_dates = unique_dates[:val_start_idx]
    val_dates = unique_dates[val_start_idx:test_start_idx]
    test_dates = unique_dates[test_start_idx:]

    print(f"\nTime-based splits:")
    print(f"Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Val:   {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")
    print(f"Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # Create boolean masks for each split
    train_mask = df.index.get_level_values('valid_datetime').isin(train_dates)
    val_mask = df.index.get_level_values('valid_datetime').isin(val_dates)
    test_mask = df.index.get_level_values('valid_datetime').isin(test_dates)

    # Apply masks to create splits
    splits = {
        'X_train': data_dict['X_combined'][train_mask],
        'X_val': data_dict['X_combined'][val_mask],
        'X_test': data_dict['X_combined'][test_mask],
        'y_train': data_dict['y'][train_mask],
        'y_val': data_dict['y'][val_mask],
        'y_test': data_dict['y'][test_mask],
        'train_dates': train_dates,
        'val_dates': val_dates,
        'test_dates': test_dates
    }

    print(f"\nSample counts:")
    print(f"Train: {len(splits['y_train'])}")
    print(f"Val:   {len(splits['y_val'])}")
    print(f"Test:  {len(splits['y_test'])}")

    return splits

def evaluate_model_performance(y_true, y_pred, model_name="Model"):
    """
    Calculate comprehensive performance metrics for model evaluation.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for reporting

    Returns:
    --------
    dict: Performance metrics including MAE, RMSE, R², and bias
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)  # Mean error (systematic bias)

    return {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'bias': bias,
        'n_samples': len(y_true)
    }

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Perform hyperparameter tuning for Random Forest model.
    
    Uses grid search with cross-validation to find optimal hyperparameters.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_val, y_val : Validation data (unused, kept for compatibility)

    Returns:
    --------
    dict: Best model
    None: Placeholder for scaler (no longer needed)
    """
    print("Starting hyperparameter tuning...")

    # Random Forest parameter grid
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    best_models = {}

    # Tune Random Forest
    print("\nTuning Random Forest...")
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(
        rf_base, rf_param_grid,
        cv=3, scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train, y_train)

    best_models['Random Forest'] = rf_grid.best_estimator_
    print(f"Best RF params: {rf_grid.best_params_}")
    print(f"Best RF CV score: {-rf_grid.best_score_:.3f}")

    return best_models, None

def train_postprocessing_models_with_tuning(splits, data_dict, target_config=None):
    """
    Train tuned post-processing models and evaluate their performance.
    
    This is the main training function that coordinates hyperparameter tuning,
    model training, and performance evaluation.

    Parameters:
    -----------
    splits : dict
        Train/validation/test splits from create_time_based_splits
    data_dict : dict
        Data dictionary from prepare_postprocessing_data
    target_config : dict or None
        Target configuration dict from get_target_config(). If None, uses current TARGET_VARIABLE.

    Returns:
    --------
    dict: Complete results including trained models, performance metrics, and data
    """
    if target_config is None:
        target_config = get_target_config()
        
    # Use the actual column names that exist in the dataset (with prefixes)
    raw_forecast_col = 'gefs_' + target_config['forecast_col']  # gefs_tmax_2m
    nbm_forecast_col = 'nbm_' + target_config['forecast_col']  # nbm_tmax_2m
    # Prepare clean training and validation data
    X_train, X_val = splits['X_train'], splits['X_val']
    y_train, y_val = splits['y_train'], splits['y_val']

    # Remove any remaining NaN values
    train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    val_mask = ~(X_val.isnull().any(axis=1) | y_val.isnull())

    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]
    X_val_clean = X_val[val_mask]
    y_val_clean = y_val[val_mask]

    print(f"Clean training samples: {len(y_train_clean)}")
    print(f"Clean validation samples: {len(y_val_clean)}")

    # Perform hyperparameter tuning
    best_models, _ = tune_hyperparameters(X_train_clean, y_train_clean, X_val_clean, y_val_clean)

    # Train final models and evaluate on validation set
    results = {}
    trained_models = {}

    print(f"\nEvaluating tuned models on validation set...")

    # Random Forest evaluation
    rf_model = best_models['Random Forest']
    rf_val_pred = rf_model.predict(X_val_clean)
    results['Random Forest'] = evaluate_model_performance(y_val_clean, rf_val_pred, 'Random Forest')
    trained_models['Random Forest'] = rf_model

    # Add baseline performance from existing forecast models
    forecast_cols = data_dict['forecast_cols']

    # Always evaluate NBM model for baseline comparison if available
    # (regardless of whether it's used as a feature in post-processing)
    # Access from the full dataframe which contains all columns
    df_full = data_dict['df_full']
    if nbm_forecast_col in df_full.columns:
        # Get NBM predictions for the validation indices
        val_indices = y_val_clean.index
        nbm_val_predictions = df_full.loc[val_indices, nbm_forecast_col]
        
        # Remove any NaN pairs
        valid_mask = ~(np.isnan(nbm_val_predictions) | np.isnan(y_val_clean))
        if valid_mask.sum() > 0:
            results['NBM Model'] = evaluate_model_performance(
                y_val_clean[valid_mask], nbm_val_predictions[valid_mask], 'NBM Model'
            )

    # Evaluate Raw model if available
    if raw_forecast_col in forecast_cols and raw_forecast_col in X_val_clean.columns:
        results['Raw Model'] = evaluate_model_performance(
            y_val_clean, X_val_clean[raw_forecast_col], 'Raw Model'
        )

    # Note if no forecast columns are available (pure ML approach)
    if not forecast_cols:
        print("Note: No forecast columns available - using pure ML approach from features only")

    return {
        'results': results,
        'models': trained_models,
        'scaler': None,  # No longer needed
        'X_train_clean': X_train_clean,
        'y_train_clean': y_train_clean,
        'X_val_clean': X_val_clean,
        'y_val_clean': y_val_clean,
        'forecast_cols': forecast_cols
    }

# =============================================================================
# ANALYSIS AND VISUALIZATION FUNCTIONS
# =============================================================================

def analyze_feature_importance(model_results, data_dict):
    """
    Analyze and visualize feature importance from the Random Forest model.
    
    Feature importance helps understand which meteorological variables
    are most influential for temperature prediction.

    Parameters:
    -----------
    model_results : dict
        Results from train_postprocessing_models_with_tuning
    data_dict : dict
        Data dictionary with feature information

    Returns:
    --------
    pd.DataFrame: Feature importance rankings
    """
    rf_model = model_results['models']['Random Forest']
    feature_names = model_results['X_train_clean'].columns

    # Extract feature importances from trained Random Forest
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Create plots directory if it doesn't exist
    plots_dir = Path('./plots')
    plots_dir.mkdir(exist_ok=True)

    # Create feature importance visualization
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)  # Show top 20 features

    plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance (Top 20)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print top features for analysis
    print("Top 10 Most Important Features:")
    print("=" * 40)
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"{row['feature']:<25} {row['importance']:.4f}")

    return feature_importance_df

def plot_forecast_vs_observed_comparison(model_results, data_dict):
    """
    Create side-by-side scatter plots comparing NBM and Random Forest forecasts against observations.
    
    Parameters:
    -----------
    model_results : dict
        Results from model training
    data_dict : dict
        Data dictionary with forecast information
    """
    results = model_results['results']
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('./plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Get validation data
    X_val_clean = model_results['X_val_clean']
    y_val_clean = model_results['y_val_clean']
    
    # Generate Random Forest predictions
    rf_predictions = model_results['models']['Random Forest'].predict(X_val_clean)
    
    # Get NBM predictions (always check full dataframe, not just features)
    nbm_col = None
    df_full = data_dict['df_full']
    
    # Look for NBM column in the full dataframe
    for col in df_full.columns:
        if 'nbm_' in col and ('tmax_2m' in col or 'tmin_2m' in col):
            nbm_col = col
            break
    
    # Create the comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Common plot settings
    point_size = 15  # Larger points for visibility
    alpha = 0.7
    
    # Left subplot: NBM vs Observed
    if nbm_col and nbm_col in df_full.columns:
        # Get NBM predictions from the full dataframe for validation indices
        val_indices = y_val_clean.index
        nbm_predictions = df_full.loc[val_indices, nbm_col]
        
        # Remove any NaN pairs
        valid_mask = ~(np.isnan(nbm_predictions) | np.isnan(y_val_clean))
        nbm_clean = nbm_predictions[valid_mask]
        y_nbm_clean = y_val_clean[valid_mask]
        
        ax1.scatter(y_nbm_clean, nbm_clean, s=point_size, alpha=alpha, color='lightblue', edgecolors='blue', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_nbm_clean.min(), nbm_clean.min())
        max_val = max(y_nbm_clean.max(), nbm_clean.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate metrics
        nbm_r2 = results['NBM Model']['r2'] if 'NBM Model' in results else r2_score(y_nbm_clean, nbm_clean)
        nbm_mae = results['NBM Model']['mae'] if 'NBM Model' in results else mean_absolute_error(y_nbm_clean, nbm_clean)
        nbm_rmse = results['NBM Model']['rmse'] if 'NBM Model' in results else np.sqrt(mean_squared_error(y_nbm_clean, nbm_clean))
        nbm_count = len(y_nbm_clean)
        
        # Add metrics text
        metrics_text = f'R² = {nbm_r2:.3f}\nMAE = {nbm_mae:.3f}°C\nRMSE = {nbm_rmse:.3f}°C\nN = {nbm_count:,}'
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Observed Temperature (°C)')
        ax1.set_ylabel('NBM Forecast (°C)')
        ax1.set_title('NBM Model vs Observed')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'NBM data not available', transform=ax1.transAxes, 
                ha='center', va='center', fontsize=14)
        ax1.set_title('NBM Model vs Observed (No Data)')
    
    # Right subplot: Random Forest vs Observed
    # Remove any NaN pairs
    valid_mask = ~(np.isnan(rf_predictions) | np.isnan(y_val_clean))
    rf_clean = rf_predictions[valid_mask]
    y_rf_clean = y_val_clean[valid_mask]
    
    ax2.scatter(y_rf_clean, rf_clean, s=point_size, alpha=alpha, color='lightgreen', edgecolors='darkgreen', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_rf_clean.min(), rf_clean.min())
    max_val = max(y_rf_clean.max(), rf_clean.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics
    rf_r2 = results['Random Forest']['r2']
    rf_mae = results['Random Forest']['mae']
    rf_rmse = results['Random Forest']['rmse']
    rf_count = len(y_rf_clean)
    
    # Add metrics text
    metrics_text = f'R² = {rf_r2:.3f}\nMAE = {rf_mae:.3f}°C\nRMSE = {rf_rmse:.3f}°C\nN = {rf_count:,}'
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Observed Temperature (°C)')
    ax2.set_ylabel('Random Forest Forecast (°C)')
    ax2.set_title('Random Forest vs Observed')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Make axes equal and aligned
    if nbm_col and nbm_col in df_full.columns:
        # Get global min/max for consistent scaling
        all_obs = np.concatenate([y_nbm_clean, y_rf_clean])
        all_pred = np.concatenate([nbm_clean, rf_clean])
        global_min = min(all_obs.min(), all_pred.min())
        global_max = max(all_obs.max(), all_pred.max())
        
        ax1.set_xlim(global_min, global_max)
        ax1.set_ylim(global_min, global_max)
        ax2.set_xlim(global_min, global_max)
        ax2.set_ylim(global_min, global_max)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
    else:
        ax2.set_xlim(y_rf_clean.min(), y_rf_clean.max())
        ax2.set_ylim(rf_clean.min(), rf_clean.max())
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plots_dir / 'forecast_vs_observed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison_enhanced(model_results, data_dict):
    """
    Create comprehensive comparison plots of model performance.
    
    Includes performance metrics, scatter plots, residual analysis,
    and improvement calculations relative to baseline models.

    Parameters:
    -----------
    model_results : dict
        Results from model training
    data_dict : dict
        Data dictionary with forecast information
    """
    results = model_results['results']

    # Display performance comparison table
    print("\nModel Performance Comparison (Validation Set):")
    print("=" * 60)
    print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Bias':<8}")
    print("-" * 60)
    for model_name in ['NBM Model', 'Random Forest']:
        if model_name in results:
            r = results[model_name]
            print(f"{model_name:<15} {r['mae']:<8.3f} {r['rmse']:<8.3f} {r['r2']:<8.3f} {r['bias']:<8.3f}")

    # Create plots directory if it doesn't exist
    plots_dir = Path('./plots')
    plots_dir.mkdir(exist_ok=True)

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    model_names = ['NBM Model', 'Random Forest']
    colors = ['lightblue', 'lightgreen']

    # Filter model names to only those available in results
    available_models = [name for name in model_names if name in results]
    available_colors = [colors[model_names.index(name)] for name in available_models]

    # Performance metric comparisons
    # MAE comparison
    mae_values = [results[m]['mae'] for m in available_models]
    axes[0,0].bar(available_models, mae_values, color=available_colors)
    axes[0,0].set_title('Mean Absolute Error (MAE)')
    axes[0,0].set_ylabel('MAE (°C)')
    axes[0,0].tick_params(axis='x', rotation=45)

    # RMSE comparison
    rmse_values = [results[m]['rmse'] for m in available_models]
    axes[0,1].bar(available_models, rmse_values, color=available_colors)
    axes[0,1].set_title('Root Mean Square Error (RMSE)')
    axes[0,1].set_ylabel('RMSE (°C)')
    axes[0,1].tick_params(axis='x', rotation=45)

    # R² comparison
    r2_values = [results[m]['r2'] for m in available_models]
    axes[0,2].bar(available_models, r2_values, color=available_colors)
    axes[0,2].set_title('R² Score')
    axes[0,2].set_ylabel('R²')
    axes[0,2].tick_params(axis='x', rotation=45)

    # Bias comparison
    bias_values = [results[m]['bias'] for m in available_models]
    axes[1,0].bar(available_models, bias_values, color=available_colors)
    axes[1,0].set_title('Bias (Mean Error)')
    axes[1,0].set_ylabel('Bias (°C)')
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].tick_params(axis='x', rotation=45)

    # Scatter plot for Random Forest model
    ml_models = [name for name in available_models if name != 'NBM Model']
    if ml_models:
        # Use Random Forest (the only ML model now)
        y_pred = model_results['models']['Random Forest'].predict(model_results['X_val_clean'])

        y_true = model_results['y_val_clean']
        axes[1,1].scatter(y_true, y_pred, alpha=0.6, s=1)
        axes[1,1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1,1].set_xlabel('Observed Temperature (°C)')
        axes[1,1].set_ylabel('Predicted Temperature (°C)')
        axes[1,1].set_title('Random Forest: Observed vs Predicted')

        # Residual plot for Random Forest
        residuals = y_pred - y_true
        axes[1,2].scatter(y_pred, residuals, alpha=0.6, s=1)
        axes[1,2].axhline(y=0, color='r', linestyle='--')
        axes[1,2].set_xlabel('Predicted Temperature (°C)')
        axes[1,2].set_ylabel('Residuals (°C)')
        axes[1,2].set_title('Random Forest: Residual Plot')

    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate and display improvement over NBM baseline
    if 'NBM Model' in results:
        print(f"\nImprovement over NBM Model:")
        if 'Random Forest' in results:
            mae_improvement = ((results['NBM Model']['mae'] - results['Random Forest']['mae']) /
                             results['NBM Model']['mae']) * 100
            rmse_improvement = ((results['NBM Model']['rmse'] - results['Random Forest']['rmse']) /
                              results['NBM Model']['rmse']) * 100
            print(f"Random Forest: MAE {mae_improvement:+.1f}%, RMSE {rmse_improvement:+.1f}%")

# =============================================================================
# MULTI-FORECAST HOUR ANALYSIS FUNCTIONS
# =============================================================================

def evaluate_metrics_by_forecast_hour(forecast_hours=None, target_config=None):
    """
    Evaluate model performance metrics (R², MAE, RMSE) across multiple forecast hours.
    
    Parameters:
    -----------
    forecast_hours : List[str] or None
        List of forecast hours to evaluate. If None, uses FORECAST_HOURS from config.
    target_config : dict or None
        Target configuration dict from get_target_config(). If None, uses current TARGET_VARIABLE.
        
    Returns:
    --------
    dict: Results containing metrics for each forecast hour and plotting data
    """
    if forecast_hours is None:
        forecast_hours = FORECAST_HOURS
    if target_config is None:
        target_config = get_target_config()
    
    results_by_hour = {}
    metrics_df_list = []
    
    print(f"=== EVALUATING METRICS ACROSS {len(forecast_hours)} FORECAST HOURS ===")
    
    for i, fhour in enumerate(forecast_hours):
        print(f"\n--- Processing {fhour} ({i+1}/{len(forecast_hours)}) ---")
        
        try:
            # Run the main pipeline for this forecast hour - pass specific forecast hour
            pipeline_results = run_single_forecast_hour_pipeline(target_config, forecast_hour=fhour)
            
            # Extract performance metrics
            model_results = pipeline_results['results']['results']
            
            # Collect metrics for each model
            for model_name, metrics in model_results.items():
                metrics_row = {
                    'forecast_hour': fhour,
                    'forecast_hour_numeric': int(fhour[1:]),  # Convert f024 -> 24
                    'model': model_name,
                    'r2': metrics['r2'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'bias': metrics['bias'],
                    'n_samples': metrics['n_samples']
                }
                metrics_df_list.append(metrics_row)
            
            results_by_hour[fhour] = pipeline_results
            print(f"✓ {fhour} completed successfully")
            
        except Exception as e:
            print(f"✗ Error processing {fhour}: {str(e)}")
            results_by_hour[fhour] = {'error': str(e)}
    
    # Create comprehensive results DataFrame
    metrics_df = pd.DataFrame(metrics_df_list)
    
    return {
        'metrics_df': metrics_df,
        'results_by_hour': results_by_hour,
        'forecast_hours': forecast_hours
    }

def plot_metrics_timeseries(metrics_results, save_plots=True):
    """
    Create timeseries plots of model performance metrics across forecast hours.
    
    Parameters:
    -----------
    metrics_results : dict
        Results from evaluate_metrics_by_forecast_hour()
    save_plots : bool
        Whether to save plots to disk
    """
    print("=== STARTING PLOT GENERATION ===")
    
    # Debug: Check if we have the right data structure
    if not isinstance(metrics_results, dict):
        print(f"ERROR: metrics_results is not a dict, got {type(metrics_results)}")
        return
    
    if 'metrics_df' not in metrics_results:
        print(f"ERROR: 'metrics_df' not found in metrics_results. Keys: {metrics_results.keys()}")
        return
    
    metrics_df = metrics_results['metrics_df']
    
    print(f"DEBUG: metrics_df type: {type(metrics_df)}")
    print(f"DEBUG: metrics_df shape: {metrics_df.shape}")
    
    if metrics_df.empty:
        print("ERROR: No metrics data available for plotting - DataFrame is empty")
        return
    
    print(f"DEBUG: metrics_df columns: {list(metrics_df.columns)}")
    print(f"DEBUG: metrics_df sample:\n{metrics_df.head()}")
    
    # Create plots directory if needed
    if save_plots:
        plots_dir = Path('./plots')
        plots_dir.mkdir(exist_ok=True)
        print(f"DEBUG: Created plots directory: {plots_dir}")
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Metrics vs Forecast Hour', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['r2', 'mae', 'rmse', 'bias']
        metric_labels = ['R²', 'MAE (°C)', 'RMSE (°C)', 'Bias (°C)']
        
        print("DEBUG: Starting to create subplots...")
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            print(f"DEBUG: Processing metric {metric} ({label})")
            
            # Plot each model
            plotted_any = False
            for model in metrics_df['model'].unique():
                model_data = metrics_df[metrics_df['model'] == model].sort_values('forecast_hour_numeric')
                
                print(f"DEBUG: Model {model} has {len(model_data)} data points")
                
                if not model_data.empty:
                    x_vals = model_data['forecast_hour_numeric']
                    y_vals = model_data[metric]
                    
                    print(f"DEBUG: Plotting {model}: x={list(x_vals)}, y={list(y_vals)}")
                    
                    ax.plot(x_vals, y_vals, 
                           marker='o', linewidth=2, markersize=8, label=model)
                    plotted_any = True
            
            if not plotted_any:
                print(f"WARNING: No data plotted for metric {metric}")
            
            ax.set_xlabel('Forecast Hour')
            ax.set_ylabel(label)
            ax.set_title(f'{label} vs Forecast Hour')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis to show forecast hours
            forecast_hours_numeric = sorted(metrics_df['forecast_hour_numeric'].unique())
            ax.set_xticks(forecast_hours_numeric)
            ax.set_xticklabels([f'f{h:03d}' for h in forecast_hours_numeric], rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = plots_dir / 'metrics_vs_forecast_hour.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: Plot saved to: {plot_path}")
        
        print("DEBUG: Attempting to show plot...")
        plt.show()
        print("DEBUG: Plot shown successfully")
        
        # Print summary statistics
        print("\n=== PERFORMANCE SUMMARY ACROSS FORECAST HOURS ===")
        for model in metrics_df['model'].unique():
            model_data = metrics_df[metrics_df['model'] == model]
            print(f"\n{model}:")
            print(f"  R² range: {model_data['r2'].min():.3f} - {model_data['r2'].max():.3f}")
            print(f"  MAE range: {model_data['mae'].min():.3f} - {model_data['mae'].max():.3f} °C")
            print(f"  RMSE range: {model_data['rmse'].min():.3f} - {model_data['rmse'].max():.3f} °C")
            print(f"  Bias range: {model_data['bias'].min():.3f} - {model_data['bias'].max():.3f} °C")
            
    except Exception as e:
        print(f"ERROR in plot generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("=== PLOT GENERATION COMPLETED ===")
    return True

def run_single_forecast_hour_pipeline(target_config, forecast_hour=None):
    """
    Run the ML pipeline for a single forecast hour (used internally by evaluate_metrics_by_forecast_hour).
    This is essentially the same as the main() function but returns results instead of printing summary.
    
    Parameters:
    -----------
    target_config : dict
        Target configuration dict from get_target_config()
    forecast_hour : str or None
        Single forecast hour to process (e.g., 'f024'). If None, uses all FORECAST_HOURS.
    """
    # Determine which forecast hours to use
    if forecast_hour is not None:
        forecast_hours_to_use = [forecast_hour]
        print(f"Processing single forecast hour: {forecast_hour}")
    else:
        forecast_hours_to_use = FORECAST_HOURS
        print(f"Processing all forecast hours: {forecast_hours_to_use}")
    
    # Step 1: Load and combine data for specific forecast hour(s)
    forecast, nbm, urma = load_combined_data(
        station_ids=STATION_IDS,
        forecast_hours=forecast_hours_to_use,
        base_path=BASE_PATH
    )

    # Step 2: Prepare data subsets with proper column naming
    forecast_cols = [target_config['forecast_col'], target_config['obs_col']]
    nbm_cols = [target_config['forecast_col'], target_config['obs_col']]
    
    forecast_subset = forecast[forecast_cols].dropna().sort_index()
    nbm_subset = nbm[nbm_cols].dropna().sort_index()
    urma_subset = urma[[target_config['urma_field']]].dropna()

    # Standardize URMA column names and index
    urma_subset = urma_subset.reset_index().rename(
        columns={target_config['urma_field']: target_config['obs_col'],
                 'valid_time': 'valid_datetime', 'station_id': 'sid'}
    ).set_index(['valid_datetime', 'sid']).sort_index()

    # Add prefixes to distinguish data sources
    forecast_subset.columns = ['gefs_' + col for col in forecast_subset.columns]
    nbm_subset.columns = ['nbm_' + col for col in nbm_subset.columns]
    urma_subset.columns = ['urma_' + col for col in urma_subset.columns]

    # Convert URMA temperature from Kelvin to Celsius
    urma_subset[target_config['urma_obs_col']] -= 273.15

    # Step 3: Create time-matched dataset
    time_matched_df = create_time_matched_dataset(forecast_subset, nbm_subset, urma_subset)

    # Step 4: Apply quality control
    time_matched_qc, qc_stats = apply_qc_filter(time_matched_df, target_config=target_config)

    # Step 5: Prepare ML dataset
    target_data = time_matched_qc[time_matched_qc[target_config['nbm_col']].notna()].reset_index(
    ).set_index(['valid_datetime', 'sid']).sort_index()
    
    target_data.drop(columns=[target_config['nbm_obs_col']], inplace=True)

    # Clean feature data
    feature_data, drop_info = identify_and_drop_non_predictive_columns(
        forecast[forecast[target_config['obs_col']].notna()].sort_index())

    # Combine features and targets
    target_data_with_both = target_data.copy()
    if target_config['nbm_col'] in target_data.columns:
        ml_dataset, merge_stats = combine_features_and_targets(
            feature_data, target_data_with_both, target_config=target_config
        )
        common_indices = feature_data.index.intersection(target_data.index)
        if target_config['nbm_col'] in target_data.columns:
            ml_dataset[target_config['nbm_col']] = target_data.loc[common_indices, target_config['nbm_col']]
    else:
        ml_dataset, merge_stats = combine_features_and_targets(
            feature_data, target_data_with_both, target_config=target_config
        )

    # Remove unnecessary columns
    opposite_type = 'tmin_2m' if target_config['target_type'] == 'tmax' else 'tmax_2m'
    if opposite_type in ml_dataset.columns:
        ml_dataset.drop(columns=[opposite_type], inplace=True)

    # Step 6: Prepare data for post-processing
    data_dict = prepare_postprocessing_data(ml_dataset, target_config=target_config)

    # Step 7: Create time-based splits
    splits = create_time_based_splits(data_dict)

    # Step 8: Train and evaluate models
    model_results = train_postprocessing_models_with_tuning(splits, data_dict, target_config=target_config)

    return {
        'data': {
            'forecast': forecast,
            'nbm': nbm,
            'urma': urma,
            'time_matched_qc': time_matched_qc,
            'ml_dataset': ml_dataset
        },
        'results': model_results,
        'qc_stats': qc_stats,
        'splits': splits
    }

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """
    Main execution function that orchestrates the complete ML pipeline.
    """
    # Get target configuration
    target_config = get_target_config()
    
    print("=== GEFS ML POST-PROCESSING PIPELINE ===")
    print(f"Base path: {BASE_PATH}")
    print(f"Stations: {STATION_IDS}")
    print(f"Forecast hours: {FORECAST_HOURS}")
    print(f"QC threshold: {QC_THRESHOLD}°C")
    print(f"Target variable: {target_config['description']} ({target_config['target_type']})")
    print(f"Mode: {'Post-processing NBM' if POSTPROCESS_NBM else 'Independent forecasting (no NBM)'}")
    
    # Step 1: Load and combine data
    print("\n=== STEP 1: LOADING DATA ===")
    forecast, nbm, urma = load_combined_data()

    # Step 2: Prepare data subsets with proper column naming
    forecast_cols = [target_config['forecast_col'], target_config['obs_col']]
    nbm_cols = [target_config['forecast_col'], target_config['obs_col']]
    
    forecast_subset = forecast[forecast_cols].dropna().sort_index()
    nbm_subset = nbm[nbm_cols].dropna().sort_index()
    urma_subset = urma[[target_config['urma_field']]].dropna()

    # Standardize URMA column names and index
    urma_subset = urma_subset.reset_index().rename(
        columns={target_config['urma_field']: target_config['obs_col'],
                 'valid_time': 'valid_datetime', 'station_id': 'sid'}
    ).set_index(['valid_datetime', 'sid']).sort_index()

    # Add prefixes to distinguish data sources
    forecast_subset.columns = ['gefs_' + col for col in forecast_subset.columns]
    nbm_subset.columns = ['nbm_' + col for col in nbm_subset.columns]
    urma_subset.columns = ['urma_' + col for col in urma_subset.columns]

    # Convert URMA temperature from Kelvin to Celsius
    urma_subset[target_config['urma_obs_col']] -= 273.15

    # Step 3: Create time-matched dataset
    print("\n=== STEP 2: CREATING TIME-MATCHED DATASET ===")
    time_matched_df = create_time_matched_dataset(forecast_subset, nbm_subset, urma_subset)
    print(f"Time-matched dataset shape: {time_matched_df.shape}")

    # Step 4: Apply quality control
    print("\n=== STEP 3: APPLYING QUALITY CONTROL ===")
    time_matched_qc, qc_stats = apply_qc_filter(time_matched_df, target_config=target_config)
    
    print(f"QC Results:")
    print(f"- Threshold: {qc_stats['threshold_used']}°C")
    print(f"- Records passing QC: {qc_stats['qc_passed']:,}/{qc_stats['records_with_both_obs_urma']:,}")
    print(f"- QC pass rate: {qc_stats['qc_pass_rate']*100:.1f}%")
    print(f"- Final dataset shape: {time_matched_qc.shape}")

    # Step 5: Prepare ML dataset
    print("\n=== STEP 4: PREPARING ML DATASET ===")
    
    # Create target dataset from QC'd data
    target_data = time_matched_qc[time_matched_qc[target_config['nbm_col']].notna()].reset_index(
    ).set_index(['valid_datetime', 'sid']).sort_index()
    
    # Keep the GEFS observations as the main target, drop NBM observations
    target_data.drop(columns=[target_config['nbm_obs_col']], inplace=True)
    # The gefs_obs_col will be used as the target, no need to rename it

    # Clean feature data
    feature_data, drop_info = identify_and_drop_non_predictive_columns(
        forecast[forecast[target_config['obs_col']].notna()].sort_index())
    
    print(f"Feature cleaning results:")
    print(f"- Original shape: {drop_info['original_shape']}")
    print(f"- Cleaned shape: {drop_info['cleaned_shape']}")
    print(f"- Dropped {len(drop_info['dropped_columns'])} columns")

    # Combine features and targets - always include NBM for baseline comparison
    target_data_with_both = target_data.copy()
    if target_config['nbm_col'] in target_data.columns:
        # Both columns are already present, we can use them directly
        ml_dataset, merge_stats = combine_features_and_targets(
            feature_data, target_data_with_both, target_config=target_config
        )
        # Manually add the NBM column since combine_features_and_targets only handles one target
        common_indices = feature_data.index.intersection(target_data.index)
        if target_config['nbm_col'] in target_data.columns:
            ml_dataset[target_config['nbm_col']] = target_data.loc[common_indices, target_config['nbm_col']]
    else:
        ml_dataset, merge_stats = combine_features_and_targets(
            feature_data, target_data_with_both, target_config=target_config
        )

    # Remove unnecessary columns - drop the opposite temperature type
    opposite_type = 'tmin_2m' if target_config['target_type'] == 'tmax' else 'tmax_2m'
    if opposite_type in ml_dataset.columns:
        ml_dataset.drop(columns=[opposite_type], inplace=True)

    print(f"Final ML dataset shape: {ml_dataset.shape}")
    print(f"Final columns: {len(ml_dataset.columns)}")

    # Step 6: Prepare data for post-processing
    print("\n=== STEP 5: PREPARING POST-PROCESSING DATA ===")
    data_dict = prepare_postprocessing_data(ml_dataset, target_config=target_config)

    # Step 7: Create time-based splits
    print("\n=== STEP 6: CREATING TIME-BASED SPLITS ===")
    splits = create_time_based_splits(data_dict)

    # Step 8: Train and evaluate models
    print("\n=== STEP 7: TRAINING AND EVALUATING MODELS ===")
    model_results = train_postprocessing_models_with_tuning(splits, data_dict, target_config=target_config)

    # Step 9: Analyze results
    print("\n=== STEP 8: ANALYZING RESULTS ===")
    
    # Feature importance analysis
    print("\n--- Feature Importance Analysis ---")
    feature_importance_df = analyze_feature_importance(model_results, data_dict)

    # Forecast vs observed comparison
    print("\n--- Forecast vs Observed Comparison ---")
    plot_forecast_vs_observed_comparison(model_results, data_dict)

    # Performance comparison and visualization
    print("\n--- Performance Comparison ---")
    plot_model_comparison_enhanced(model_results, data_dict)

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    
    return {
        'data': {
            'forecast': forecast,
            'nbm': nbm,
            'urma': urma,
            'time_matched_qc': time_matched_qc,
            'ml_dataset': ml_dataset
        },
        'results': model_results,
        'feature_importance': feature_importance_df,
        'qc_stats': qc_stats,
        'splits': splits
    }

def main_multi_forecast_hour():
    """
    Main execution function for multi-forecast hour analysis.
    Evaluates model performance across all forecast hours and creates timeseries plots.
    """
    print("=== MULTI-FORECAST HOUR ANALYSIS ===")
    print(f"Evaluating forecast hours: {FORECAST_HOURS}")
    
    # Run evaluation across all forecast hours
    metrics_results = evaluate_metrics_by_forecast_hour()
    
    # Create timeseries plots
    print("\n=== CREATING TIMESERIES PLOTS ===")
    plot_metrics_timeseries(metrics_results)
    
    print("\n=== MULTI-FORECAST HOUR ANALYSIS COMPLETED ===")
    return metrics_results

def load_and_plot_existing_results():
    """
    Load existing multi-forecast hour results and create timeseries plots.
    Useful when you want to recreate plots without rerunning the analysis.
    """
    import pickle
    
    results_file = Path('./results/multi_forecast_hour_results.pkl')
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run the main script with MULTI_FORECAST_HOUR_ANALYSIS = True first")
        return None
    
    print(f"Loading existing results from: {results_file}")
    with open(results_file, 'rb') as f:
        metrics_results = pickle.load(f)
    
    # Debug the loaded results
    print("=== DEBUGGING LOADED RESULTS ===")
    print(f"Type of loaded data: {type(metrics_results)}")
    print(f"Keys in loaded data: {metrics_results.keys() if isinstance(metrics_results, dict) else 'Not a dict'}")
    
    if 'metrics_df' in metrics_results:
        metrics_df = metrics_results['metrics_df']
        print(f"metrics_df type: {type(metrics_df)}")
        print(f"metrics_df shape: {metrics_df.shape}")
        print(f"metrics_df empty: {metrics_df.empty}")
        if not metrics_df.empty:
            print(f"metrics_df columns: {list(metrics_df.columns)}")
            print(f"metrics_df head:\n{metrics_df.head()}")
    
    if 'results_by_hour' in metrics_results:
        results_by_hour = metrics_results['results_by_hour']
        print(f"\nresults_by_hour type: {type(results_by_hour)}")
        print(f"results_by_hour keys: {list(results_by_hour.keys()) if isinstance(results_by_hour, dict) else 'Not a dict'}")
        
        # Check for errors in each forecast hour
        for fhour, result in results_by_hour.items():
            if isinstance(result, dict) and 'error' in result:
                print(f"ERROR in {fhour}: {result['error']}")
            elif isinstance(result, dict):
                print(f"{fhour}: Success (keys: {list(result.keys())})")
            else:
                print(f"{fhour}: Unexpected result type: {type(result)}")
    
    # Extract the metrics DataFrame
    metrics_df = metrics_results.get('metrics_df', pd.DataFrame())
    
    if metrics_df.empty:
        print("No metrics data found in results")
        print("This suggests all forecast hours failed during processing.")
        print("Check the error messages above for details.")
        return None
    
    print(f"Loaded metrics for {len(metrics_df)} model-forecast hour combinations")
    print(f"Models: {metrics_df['model'].unique()}")
    print(f"Forecast hours: {sorted(metrics_df['forecast_hour_numeric'].unique())}")
    
    # Create the timeseries plots
    print("\n=== CREATING TIMESERIES PLOTS FROM SAVED RESULTS ===")
    plot_metrics_timeseries(metrics_results)
    
    # Print detailed summary
    print("\n=== DETAILED METRICS TABLE ===")
    pivot_table = metrics_df.pivot_table(index='forecast_hour', 
                                        columns='model', 
                                        values=['r2', 'mae', 'rmse'], 
                                        aggfunc='first').round(3)
    print(pivot_table)
    
    return metrics_results

def debug_single_forecast_hour(fhour='f024'):
    """
    Debug function to test a single forecast hour and see what's failing.
    """
    print(f"=== DEBUGGING SINGLE FORECAST HOUR: {fhour} ===")
    
    # Get target configuration
    target_config = get_target_config()
    print(f"Target config: {target_config}")
    
    try:
        print(f"Step 1: Loading data for {fhour}")
        forecast, nbm, urma = load_combined_data(
            station_ids=STATION_IDS,
            forecast_hours=[fhour],  # Only load the specific forecast hour
            base_path=BASE_PATH
        )
        print(f"Loaded - Forecast: {forecast.shape}, NBM: {nbm.shape}, URMA: {urma.shape}")
        
        print(f"Step 2: Preparing data subsets")
        forecast_cols = [target_config['forecast_col'], target_config['obs_col']]
        nbm_cols = [target_config['forecast_col'], target_config['obs_col']]
        
        print(f"Looking for forecast columns: {forecast_cols}")
        print(f"Available forecast columns: {list(forecast.columns)}")
        print(f"Looking for NBM columns: {nbm_cols}")
        print(f"Available NBM columns: {list(nbm.columns)}")
        
        # Check if required columns exist
        missing_forecast_cols = [col for col in forecast_cols if col not in forecast.columns]
        missing_nbm_cols = [col for col in nbm_cols if col not in nbm.columns]
        
        if missing_forecast_cols:
            print(f"ERROR: Missing forecast columns: {missing_forecast_cols}")
        if missing_nbm_cols:
            print(f"ERROR: Missing NBM columns: {missing_nbm_cols}")
            
        if missing_forecast_cols or missing_nbm_cols:
            print("Cannot continue - missing required columns")
            return
        
        forecast_subset = forecast[forecast_cols].dropna().sort_index()
        nbm_subset = nbm[nbm_cols].dropna().sort_index()
        print(f"Subsets - Forecast: {forecast_subset.shape}, NBM: {nbm_subset.shape}")
        
        # Check URMA data
        print(f"Step 3: Checking URMA data")
        print(f"URMA columns: {list(urma.columns)}")
        print(f"Looking for URMA field: {target_config['urma_field']}")
        
        if target_config['urma_field'] not in urma.columns:
            print(f"ERROR: URMA field '{target_config['urma_field']}' not found")
            print(f"Available URMA fields: {list(urma.columns)}")
            return
            
        urma_subset = urma[[target_config['urma_field']]].dropna()
        print(f"URMA subset: {urma_subset.shape}")
        
        # Test the full pipeline
        print(f"Step 4: Testing full pipeline for {fhour}")
        pipeline_results = run_single_forecast_hour_pipeline(target_config, forecast_hour=fhour)
        
        print(f"✓ Pipeline completed successfully!")
        print(f"Results keys: {list(pipeline_results.keys())}")
        
        if 'results' in pipeline_results and 'results' in pipeline_results['results']:
            model_results = pipeline_results['results']['results']
            print(f"Model results:")
            for model_name, metrics in model_results.items():
                print(f"  {model_name}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")
        
        return pipeline_results
        
    except Exception as e:
        print(f"ERROR in debug: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_multi_forecast_hour_simple():
    """
    Simple test function to run just 2 forecast hours and see what happens.
    """
    print("=== TESTING MULTI-FORECAST HOUR (SIMPLE) ===")
    
    target_config = get_target_config()
    test_hours = ['f024', 'f048']  # Just test 2 hours
    
    results = []
    
    for fhour in test_hours:
        print(f"\n--- Testing {fhour} ---")
        try:
            pipeline_result = run_single_forecast_hour_pipeline(target_config, forecast_hour=fhour)
            
            if 'results' in pipeline_result and 'results' in pipeline_result['results']:
                model_results = pipeline_result['results']['results']
                
                for model_name, metrics in model_results.items():
                    result_row = {
                        'forecast_hour': fhour,
                        'forecast_hour_numeric': int(fhour[1:]),
                        'model': model_name,
                        'r2': metrics['r2'],
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'bias': metrics['bias'],
                        'n_samples': metrics['n_samples']
                    }
                    results.append(result_row)
                    print(f"✓ {model_name}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}")
            else:
                print(f"✗ No model results found for {fhour}")
                
        except Exception as e:
            print(f"✗ Error in {fhour}: {str(e)}")
    
    if results:
        print(f"\n=== COLLECTED {len(results)} RESULTS ===")
        results_df = pd.DataFrame(results)
        print(results_df)
        
        # Save results
        results_dir = Path('./results')
        results_dir.mkdir(exist_ok=True)
        results_df.to_csv(results_dir / 'test_multi_forecast_results.csv', index=False)
        print(f"Test results saved to: {results_dir / 'test_multi_forecast_results.csv'}")
        
        return results_df
    else:
        print("No results collected - all forecast hours failed")
        return None

if __name__ == "__main__":
    # Configuration for execution mode
    # Options:
    # - 'multi_analysis': Run analysis across each forecast hour individually and create timeseries plots
    # - 'plot_existing': Load existing results and create plots without rerunning analysis  
    # - 'standard': Run the pipeline once with all forecast hours combined
    # - 'debug': Debug a single forecast hour to identify issues
    # - 'test_simple': Test just 2 forecast hours to verify the pipeline works
    
    EXECUTION_MODE = 'multi_analysis'  # Change this to control what the script does
    
    if EXECUTION_MODE == 'multi_analysis':
        # Run analysis across each forecast hour individually and create timeseries plots
        print("=== RUNNING MULTI-FORECAST HOUR ANALYSIS ===")
        print("This will evaluate each forecast hour separately and create timeseries plots.")
        print("This may take a while as it runs the full pipeline for each forecast hour.")
        
        metrics_results = main_multi_forecast_hour()
        
        # Save the results as CSV
        import pickle
        results_dir = Path('./results')
        results_dir.mkdir(exist_ok=True)
        
        # Save the metrics DataFrame as CSV
        metrics_csv_path = results_dir / 'multi_forecast_hour_metrics.csv'
        metrics_results['metrics_df'].to_csv(metrics_csv_path, index=False)
        print(f"Metrics saved to: {metrics_csv_path}")
        
        # Also save a summary CSV with pivot table format
        if not metrics_results['metrics_df'].empty:
            pivot_table = metrics_results['metrics_df'].pivot_table(
                index='forecast_hour', 
                columns='model', 
                values=['r2', 'mae', 'rmse', 'bias'], 
                aggfunc='first'
            ).round(4)
            
            summary_csv_path = results_dir / 'multi_forecast_hour_summary.csv'
            pivot_table.to_csv(summary_csv_path)
            print(f"Summary table saved to: {summary_csv_path}")
        
        # Keep the pickle for compatibility with plotting functions
        with open(results_dir / 'multi_forecast_hour_results.pkl', 'wb') as f:
            pickle.dump(metrics_results, f)
        print(f"Full results (pkl) saved to: {results_dir / 'multi_forecast_hour_results.pkl'}")
        
    elif EXECUTION_MODE == 'plot_existing':
        # Load existing results and create plots
        print("=== LOADING EXISTING RESULTS AND CREATING PLOTS ===")
        
        metrics_results = load_and_plot_existing_results()
        if metrics_results is None:
            print("No existing results found. Set EXECUTION_MODE = 'multi_analysis' to run the full analysis.")
    
    elif EXECUTION_MODE == 'debug':
        # Debug a single forecast hour to identify issues
        print("=== DEBUGGING SINGLE FORECAST HOUR ===")
        
        # First, check the existing results
        load_and_plot_existing_results()
        
        # Then debug a single forecast hour
        debug_single_forecast_hour('f024')
    
    elif EXECUTION_MODE == 'test_simple':
        # Test just 2 forecast hours to verify everything works
        print("=== TESTING SIMPLE MULTI-FORECAST HOUR ===")
        
        test_results = test_multi_forecast_hour_simple()
        if test_results is not None:
            print("✓ Test completed successfully! You can now run full analysis.")
        else:
            print("✗ Test failed - check errors above")
        
    else:  # standard
        # Execute the complete pipeline with all forecast hours combined
        print("=== RUNNING STANDARD PIPELINE ===")
        print("This will run the pipeline once with all forecast hours combined.")
        
        pipeline_results = main()
        
        # Save the results as CSV and pickle
        import pickle
        results_dir = Path('./results')
        results_dir.mkdir(exist_ok=True)
        
        # Save as pickle for full compatibility
        with open(results_dir / 'pipeline_results.pkl', 'wb') as f:
            pickle.dump(pipeline_results, f)
        print(f"Full results (pkl) saved to: {results_dir / 'pipeline_results.pkl'}")
        
        # Save key metrics as CSV if available
        if 'results' in pipeline_results and 'results' in pipeline_results['results']:
            model_metrics = []
            for model_name, metrics in pipeline_results['results']['results'].items():
                model_metrics.append({
                    'model': model_name,
                    'r2': metrics['r2'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'bias': metrics['bias'],
                    'n_samples': metrics['n_samples']
                })
            
            if model_metrics:
                metrics_df = pd.DataFrame(model_metrics)
                metrics_csv_path = results_dir / 'pipeline_model_metrics.csv'
                metrics_df.to_csv(metrics_csv_path, index=False)
                print(f"Model metrics saved to: {metrics_csv_path}")
