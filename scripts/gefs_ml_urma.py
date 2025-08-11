#!/home/chad.kahler/anaconda3/envs/dev/bin/python

"""
GEFS ML Weather Forecasting Post-Processing Pipeline

This script implements a machine learning approach to post-process weather forecasts,
specifically focusing on maximum temperature predictions. It combines GEFS (Global Ensemble
Forecast System), NBM (National Blend of Models), and URMA (Unrestricted Mesoscale Analysis)
data to train ML models that can improve forecast accuracy.

Main workflow:
1. Load and combine forecast, NBM, and URMA data
2. Create time-matched datasets with quality control
3. Prepare data for ML training
4. Train and evaluate Random Forest and Neural Network models
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
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================

# Base directory for all data files
BASE_PATH = '/nas/stid/data/gefs-ml/'

# Station IDs to include in analysis - MODIFY THIS LIST AS NEEDED
STATION_IDS = ['KSLC']

# Forecast hours to include - MODIFY THIS LIST AS NEEDED
FORECAST_HOURS = ['f024']

# Quality control threshold (maximum allowed deviation between URMA and station obs in °C)
QC_THRESHOLD = 5.0

# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================

def load_combined_data(
    station_ids: List[str] = STATION_IDS,
    forecast_hours: List[str] = FORECAST_HOURS,
    base_path: str = BASE_PATH
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and combine forecast, NBM, and URMA data into three DataFrames.
    
    This function reads CSV files for forecast and NBM data organized by forecast hour
    and station ID, and reads the URMA parquet file. All data is indexed by 
    [valid_datetime, sid/station_id] for consistent merging.

    Parameters:
    -----------
    station_ids : List[str]
        List of station IDs to load (e.g., ['KSLC', 'KBOI'])
    forecast_hours : List[str] 
        List of forecast hours to load (e.g., ['f120'])
    base_path : str
        Base directory path where data folders are located

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        forecast_df, nbm_df, urma_df with MultiIndex [valid_datetime, sid/station_id]
    """
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
    urma_df = pd.read_parquet(urma_path)

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

def apply_qc_filter(df, urma_obs_threshold=QC_THRESHOLD, obs_col='gefs_tmax_obs', urma_col='urma_tmax_obs'):
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
    obs_col : str
        Column name for station observations
    urma_col : str
        Column name for URMA analysis

    Returns:
    --------
    pd.DataFrame : QC filtered dataframe
    dict : QC statistics and diagnostics
    """
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

def combine_features_and_targets(feature_data, target_data, target_cols=None):
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
    target_cols : str, list, or None
        Target column(s) to extract

    Returns:
    --------
    pd.DataFrame : Combined dataset ready for ML
    dict : Merge statistics and information
    """
    print("=== COMBINING FEATURES AND TARGETS ===")
    print(f"Feature data shape: {feature_data.shape}")
    print(f"Target data shape: {target_data.shape}")

    # Handle target column specification
    if target_cols is None:
        target_cols = ['urma_tmax_obs']
    elif isinstance(target_cols, str):
        target_cols = [target_cols]

    # Validate target columns exist
    available_targets = [col for col in target_data.columns if any(keyword in col.lower() for keyword in ['tmax', 'temp', 'target'])]
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

def prepare_postprocessing_data(ml_dataset, target_col='tmax_obs',
                               raw_forecast_col=None,
                               nbm_forecast_col=None):
    """
    Prepare data for post-processing approach by separating features, forecasts, and targets.
    
    Post-processing uses existing forecast models as a starting point and applies ML
    to improve their predictions using additional meteorological features.

    Parameters:
    -----------
    ml_dataset : pd.DataFrame
        Combined dataset with features and targets
    target_col : str
        Target variable column name
    raw_forecast_col : str or None
        Raw model forecast column name (e.g., GEFS)
    nbm_forecast_col : str or None
        Post-processed forecast column name (e.g., NBM)

    Returns:
    --------
    dict: Contains separated datasets and baseline performance metrics
    """
    df = ml_dataset.copy()
    df = df.dropna(subset=[target_col])  # Remove rows with missing targets

    # Calculate baseline metrics for existing forecast models
    baseline_metrics = {}
    forecast_cols = []

    # Evaluate NBM forecast if available
    if nbm_forecast_col is not None and nbm_forecast_col in df.columns:
        df['nbm_bias'] = df[target_col] - df[nbm_forecast_col]
        df['nbm_abs_error'] = np.abs(df['nbm_bias'])

        nbm_mae = df['nbm_abs_error'].mean()
        nbm_rmse = np.sqrt((df['nbm_bias']**2).mean())
        baseline_metrics['nbm_mae'] = nbm_mae
        baseline_metrics['nbm_rmse'] = nbm_rmse
        forecast_cols.append(nbm_forecast_col)

        print("Baseline Performance:")
        print(f"NBM Model - MAE: {nbm_mae:.3f}°C, RMSE: {nbm_rmse:.3f}°C")

    # Evaluate raw forecast if available
    if raw_forecast_col is not None and raw_forecast_col in df.columns:
        df['raw_bias'] = df[target_col] - df[raw_forecast_col]
        df['raw_abs_error'] = np.abs(df['raw_bias'])

        raw_mae = df['raw_abs_error'].mean()
        raw_rmse = np.sqrt((df['raw_bias']**2).mean())
        baseline_metrics['raw_mae'] = raw_mae
        baseline_metrics['raw_rmse'] = raw_rmse
        forecast_cols.append(raw_forecast_col)

        if nbm_forecast_col is None:  # Only print if NBM wasn't already shown
            print("Baseline Performance:")
        print(f"Raw Model - MAE: {raw_mae:.3f}°C, RMSE: {raw_rmse:.3f}°C")

    # Separate features, forecasts, and targets
    exclude_cols = [target_col] + forecast_cols + [
        'nbm_bias', 'nbm_abs_error', 'raw_bias', 'raw_abs_error'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X_features = df[feature_cols].copy()

    # Create forecast features matrix
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
        'df_full': df,
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
    Perform hyperparameter tuning for Random Forest and Neural Network models.
    
    Uses grid search with cross-validation to find optimal hyperparameters
    for both model types.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_val, y_val : Validation data (for Neural Network scaling)

    Returns:
    --------
    dict: Best models for each algorithm
    StandardScaler: Fitted scaler for Neural Network
    """
    print("Starting hyperparameter tuning...")

    # Random Forest parameter grid
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Neural Network parameter grid
    nn_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75), (200, 100, 50)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.0001, 0.001, 0.01]  # L2 regularization
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

    # Tune Neural Network (requires feature scaling)
    print("\nTuning Neural Network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    nn_base = MLPRegressor(
        max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.1
    )
    nn_grid = GridSearchCV(
        nn_base, nn_param_grid,
        cv=3, scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=1
    )
    nn_grid.fit(X_train_scaled, y_train)

    best_models['Neural Network'] = nn_grid.best_estimator_
    print(f"Best NN params: {nn_grid.best_params_}")
    print(f"Best NN CV score: {-nn_grid.best_score_:.3f}")

    return best_models, scaler

def train_postprocessing_models_with_tuning(splits, data_dict):
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

    Returns:
    --------
    dict: Complete results including trained models, performance metrics, and data
    """
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
    best_models, scaler = tune_hyperparameters(X_train_clean, y_train_clean, X_val_clean, y_val_clean)

    # Train final models and evaluate on validation set
    results = {}
    trained_models = {}

    print(f"\nEvaluating tuned models on validation set...")

    # Random Forest evaluation
    rf_model = best_models['Random Forest']
    rf_val_pred = rf_model.predict(X_val_clean)
    results['Random Forest'] = evaluate_model_performance(y_val_clean, rf_val_pred, 'Random Forest')
    trained_models['Random Forest'] = rf_model

    # Neural Network evaluation
    nn_model = best_models['Neural Network']
    X_val_scaled = scaler.transform(X_val_clean)
    nn_val_pred = nn_model.predict(X_val_scaled)
    results['Neural Network'] = evaluate_model_performance(y_val_clean, nn_val_pred, 'Neural Network')
    trained_models['Neural Network'] = nn_model

    # Add baseline performance from existing forecast models
    forecast_cols = data_dict['forecast_cols']

    # Evaluate NBM model if available
    if 'nbm_tmax_2m' in forecast_cols and 'nbm_tmax_2m' in X_val_clean.columns:
        results['NBM Model'] = evaluate_model_performance(
            y_val_clean, X_val_clean['nbm_tmax_2m'], 'NBM Model'
        )

    # Evaluate Raw model if available
    if 'tmax_2m' in forecast_cols and 'tmax_2m' in X_val_clean.columns:
        results['Raw Model'] = evaluate_model_performance(
            y_val_clean, X_val_clean['tmax_2m'], 'Raw Model'
        )

    # Note if no forecast columns are available (pure ML approach)
    if not forecast_cols:
        print("Note: No forecast columns available - using pure ML approach from features only")

    return {
        'results': results,
        'models': trained_models,
        'scaler': scaler,
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

    # Create feature importance visualization
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)  # Show top 20 features

    plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance (Top 20)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Print top features for analysis
    print("Top 10 Most Important Features:")
    print("=" * 40)
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"{row['feature']:<25} {row['importance']:.4f}")

    return feature_importance_df

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
    for model_name in ['NBM Model', 'Random Forest', 'Neural Network']:
        if model_name in results:
            r = results[model_name]
            print(f"{model_name:<15} {r['mae']:<8.3f} {r['rmse']:<8.3f} {r['r2']:<8.3f} {r['bias']:<8.3f}")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    model_names = ['NBM Model', 'Random Forest', 'Neural Network']
    colors = ['lightblue', 'lightgreen', 'orange']

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

    # Scatter plot for best performing ML model
    ml_models = [name for name in available_models if name != 'NBM Model']
    if ml_models:
        best_model_name = min(ml_models, key=lambda x: results[x]['mae'])
        
        if best_model_name == 'Random Forest':
            y_pred = model_results['models']['Random Forest'].predict(model_results['X_val_clean'])
        else:  # Neural Network
            X_val_scaled = model_results['scaler'].transform(model_results['X_val_clean'])
            y_pred = model_results['models']['Neural Network'].predict(X_val_scaled)

        y_true = model_results['y_val_clean']
        axes[1,1].scatter(y_true, y_pred, alpha=0.6, s=1)
        axes[1,1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1,1].set_xlabel('Observed Temperature (°C)')
        axes[1,1].set_ylabel('Predicted Temperature (°C)')
        axes[1,1].set_title(f'{best_model_name}: Observed vs Predicted')

        # Residual plot for best model
        residuals = y_pred - y_true
        axes[1,2].scatter(y_pred, residuals, alpha=0.6, s=1)
        axes[1,2].axhline(y=0, color='r', linestyle='--')
        axes[1,2].set_xlabel('Predicted Temperature (°C)')
        axes[1,2].set_ylabel('Residuals (°C)')
        axes[1,2].set_title(f'{best_model_name}: Residual Plot')

    plt.tight_layout()
    plt.show()

    # Calculate and display improvement over NBM baseline
    if 'NBM Model' in results:
        print(f"\nImprovement over NBM Model:")
        for model_name in ['Random Forest', 'Neural Network']:
            if model_name in results:
                mae_improvement = ((results['NBM Model']['mae'] - results[model_name]['mae']) /
                                 results['NBM Model']['mae']) * 100
                rmse_improvement = ((results['NBM Model']['rmse'] - results[model_name]['rmse']) /
                                  results['NBM Model']['rmse']) * 100
                print(f"{model_name}: MAE {mae_improvement:+.1f}%, RMSE {rmse_improvement:+.1f}%")

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """
    Main execution function that orchestrates the complete ML pipeline.
    """
    print("=== GEFS ML POST-PROCESSING PIPELINE ===")
    print(f"Base path: {BASE_PATH}")
    print(f"Stations: {STATION_IDS}")
    print(f"Forecast hours: {FORECAST_HOURS}")
    print(f"QC threshold: {QC_THRESHOLD}°C")
    
    # Step 1: Load and combine data
    print("\n=== STEP 1: LOADING DATA ===")
    forecast, nbm, urma = load_combined_data()

    # Step 2: Prepare data subsets with proper column naming
    forecast_subset = forecast[['tmax_2m', 'tmax_obs']].dropna().sort_index()
    nbm_subset = nbm[['tmax_2m', 'tmax_obs']].dropna().sort_index()
    urma_subset = urma[['Maximum temperature_2_heightAboveGround']].dropna()

    # Standardize URMA column names and index
    urma_subset = urma_subset.reset_index().rename(
        columns={'Maximum temperature_2_heightAboveGround': 'tmax_obs',
                 'valid_time': 'valid_datetime', 'station_id': 'sid'}
    ).set_index(['valid_datetime', 'sid']).sort_index()

    # Add prefixes to distinguish data sources
    forecast_subset.columns = ['gefs_' + col for col in forecast_subset.columns]
    nbm_subset.columns = ['nbm_' + col for col in nbm_subset.columns]
    urma_subset.columns = ['urma_' + col for col in urma_subset.columns]

    # Convert URMA temperature from Kelvin to Celsius
    urma_subset['urma_tmax_obs'] -= 273.15

    # Step 3: Create time-matched dataset
    print("\n=== STEP 2: CREATING TIME-MATCHED DATASET ===")
    time_matched_df = create_time_matched_dataset(forecast_subset, nbm_subset, urma_subset)
    print(f"Time-matched dataset shape: {time_matched_df.shape}")

    # Step 4: Apply quality control
    print("\n=== STEP 3: APPLYING QUALITY CONTROL ===")
    time_matched_qc, qc_stats = apply_qc_filter(time_matched_df, urma_obs_threshold=QC_THRESHOLD)
    
    print(f"QC Results:")
    print(f"- Threshold: {qc_stats['threshold_used']}°C")
    print(f"- Records passing QC: {qc_stats['qc_passed']:,}/{qc_stats['records_with_both_obs_urma']:,}")
    print(f"- QC pass rate: {qc_stats['qc_pass_rate']*100:.1f}%")
    print(f"- Final dataset shape: {time_matched_qc.shape}")

    # Step 5: Prepare ML dataset
    print("\n=== STEP 4: PREPARING ML DATASET ===")
    
    # Create target dataset from QC'd data
    target_data = time_matched_qc[~np.isnan(time_matched_qc['nbm_tmax_2m'])].reset_index(
    ).set_index(['valid_datetime', 'sid']).sort_index()
    
    target_data.drop(columns=['nbm_tmax_obs'], inplace=True)
    target_data.rename(columns={'gefs_tmax_obs': 'tmax_obs'}, inplace=True)

    # Clean feature data
    feature_data, drop_info = identify_and_drop_non_predictive_columns(
        forecast[~np.isnan(forecast['tmax_obs'])].sort_index())
    
    print(f"Feature cleaning results:")
    print(f"- Original shape: {drop_info['original_shape']}")
    print(f"- Cleaned shape: {drop_info['cleaned_shape']}")
    print(f"- Dropped {len(drop_info['dropped_columns'])} columns")

    # Combine features and targets
    ml_dataset, merge_stats = combine_features_and_targets(
        feature_data, target_data, target_cols=['tmax_obs', 'nbm_tmax_2m']
    )

    # Remove unnecessary columns
    if 'tmin_2m' in ml_dataset.columns:
        ml_dataset.drop(columns=['tmin_2m'], inplace=True)

    print(f"Final ML dataset shape: {ml_dataset.shape}")
    print(f"Final columns: {len(ml_dataset.columns)}")

    # Step 6: Prepare data for post-processing
    print("\n=== STEP 5: PREPARING POST-PROCESSING DATA ===")
    data_dict = prepare_postprocessing_data(
        ml_dataset, 
        raw_forecast_col='tmax_2m', 
        nbm_forecast_col='nbm_tmax_2m'
    )

    # Step 7: Create time-based splits
    print("\n=== STEP 6: CREATING TIME-BASED SPLITS ===")
    splits = create_time_based_splits(data_dict)

    # Step 8: Train and evaluate models
    print("\n=== STEP 7: TRAINING AND EVALUATING MODELS ===")
    model_results = train_postprocessing_models_with_tuning(splits, data_dict)

    # Step 9: Analyze results
    print("\n=== STEP 8: ANALYZING RESULTS ===")
    
    # Feature importance analysis
    print("\n--- Feature Importance Analysis ---")
    feature_importance_df = analyze_feature_importance(model_results, data_dict)

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

if __name__ == "__main__":
    # Execute the complete pipeline
    pipeline_results = main()