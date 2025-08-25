#!/usr/bin/env python3

"""
GEFS ML Weather Forecasting Training Pipeline (Simplified)

This script implements a simplified machine learning approach to post-process weather forecasts,
focusing on temperature predictions. It combines GEFS, NBM, and URMA data to train ML models
with quality control and generates evaluation plots.

Key Features:
- URMA quality control for data validation
- Time-based train/validation/test splits
- Random Forest model training
- Single comprehensive scatter plot for evaluation

Target Variable Configuration:
- Set TARGET_VARIABLE = 'tmax' for maximum temperature predictions
- Set TARGET_VARIABLE = 'tmin' for minimum temperature predictions

Author: Simplified from gefs_ml_urma.py
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

BASE_PATH = 'data/'
STATION_IDS = ['KSLC', 'KBOI', 'KLAS', 'KSEA', 'KLAX']
FORECAST_HOURS = ['f024']
QC_THRESHOLD = 5.0  # Maximum allowed deviation between URMA and station obs in °C
TARGET_VARIABLE = 'tmax'  # 'tmax' or 'tmin'
USE_HYPERPARAMETER_TUNING = False  # Set to True for better performance but slower training
USE_VALIDATION_BASED_TRAINING = True  # Use validation set to prevent overfitting
USE_ENSEMBLE_MODELS = True  # Use multiple models and ensemble them
USE_FEATURE_SELECTION = True  # Apply feature selection to reduce overfitting
INCLUDE_NBM_AS_PREDICTOR = False  # Include NBM forecasts as input features (vs baseline only)

# =============================================================================
# TARGET VARIABLE CONFIGURATION
# =============================================================================

def get_target_config(target_type=None):
    """Get configuration for target variable based on type (tmax or tmin)."""
    if target_type is None:
        target_type = TARGET_VARIABLE
        
    if target_type.lower() in ['tmax', 'maxt']:
        return {
            'target_type': 'tmax',
            'forecast_col': 'tmax_2m',
            'obs_col': 'tmax_obs',
            'gefs_obs_col': 'gefs_tmax_obs',
            'nbm_col': 'nbm_tmax_2m',
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
            'urma_obs_col': 'urma_tmin_obs',
            'urma_field': 'Minimum temperature_2_heightAboveGround',
            'description': 'minimum temperature'
        }
    else:
        raise ValueError(f"Target type '{target_type}' not supported. Use 'tmax' or 'tmin'.")

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_combined_data():
    """Load and combine forecast, NBM, and URMA data."""
    print(f"Loading data from: {BASE_PATH}")
    print(f"Station IDs: {STATION_IDS}")
    print(f"Forecast hours: {FORECAST_HOURS}")
    
    forecast_dfs = []
    nbm_dfs = []

    # Load forecast and NBM data
    for data_type in ['forecast', 'nbm']:
        for fhour in FORECAST_HOURS:
            data_dir = Path(BASE_PATH) / data_type / fhour
            for station_id in STATION_IDS:
                # Updated file path format to match actual naming convention
                file_path = data_dir / f"{station_id}_2020_2025_{fhour}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    df['sid'] = station_id
                    if data_type == 'forecast':
                        forecast_dfs.append(df)
                    else:
                        nbm_dfs.append(df)
                else:
                    print(f"Warning: File not found: {file_path}")

    # Combine DataFrames
    forecast_df = pd.concat(forecast_dfs, ignore_index=True) if forecast_dfs else pd.DataFrame()
    nbm_df = pd.concat(nbm_dfs, ignore_index=True) if nbm_dfs else pd.DataFrame()

    # Set MultiIndex
    if not forecast_df.empty:
        forecast_df['valid_datetime'] = pd.to_datetime(forecast_df['valid_datetime'])
        forecast_df = forecast_df.set_index(['valid_datetime', 'sid'])

    if not nbm_df.empty:
        nbm_df['valid_datetime'] = pd.to_datetime(nbm_df['valid_datetime'])
        nbm_df = nbm_df.set_index(['valid_datetime', 'sid'])

    # Load URMA data
    urma_path = Path(BASE_PATH) / 'urma' / 'WR_2020_2025.urma.parquet'
    print(f"Loading URMA data from: {urma_path}")
    urma_df = pd.read_parquet(urma_path, engine='pyarrow')
    urma_df = urma_df.loc[urma_df.index.get_level_values(1).isin(STATION_IDS)]
    
    print(f"Loaded forecast data: {forecast_df.shape}")
    print(f"Loaded NBM data: {nbm_df.shape}")
    print(f"Loaded URMA data: {urma_df.shape}")

    return forecast_df, nbm_df, urma_df

def create_time_matched_dataset(forecast_subset, nbm_subset, urma_subset):
    """Create time-matched dataset with 12-hour window matching."""
    urma_times = urma_subset.index.get_level_values('valid_datetime').unique()
    stations = urma_subset.index.get_level_values('sid').unique()
    matched_data = []

    print("Creating time-matched dataset...")
    for urma_time in urma_times:
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

            # Combine matches
            for (valid_dt, sid), forecast_row in forecast_matches.iterrows():
                try:
                    nbm_row = nbm_matches.loc[(valid_dt, sid)]
                    if isinstance(nbm_row, pd.DataFrame):
                        nbm_row = nbm_row.iloc[0]
                    
                    combined_row = {
                        'urma_valid_datetime': urma_time,
                        'valid_datetime': valid_dt,
                        'sid': sid
                    }
                    combined_row.update(forecast_row.to_dict())
                    combined_row.update(nbm_row.to_dict())
                    combined_row.update(urma_data)
                    matched_data.append(combined_row)
                except KeyError:
                    continue

    if matched_data:
        result_df = pd.DataFrame(matched_data)
        result_df = result_df.set_index(['urma_valid_datetime', 'valid_datetime', 'sid'])
        print(f"Created time-matched dataset with {len(result_df)} records")
        return result_df
    else:
        print("Warning: No time-matched data created")
        return pd.DataFrame()

def apply_qc_filter(df, target_config):
    """Apply URMA quality control filter."""
    # Debug: Print available columns
    print(f"Available columns in dataset: {list(df.columns)}")
    
    obs_col = target_config['gefs_obs_col']
    urma_col = target_config['urma_obs_col']
    
    print(f"Looking for columns: '{obs_col}' and '{urma_col}'")
    
    # Check if expected columns exist, if not try to find alternatives
    if obs_col not in df.columns:
        # Try to find the column without the 'gefs_' prefix
        alt_obs_col = obs_col.replace('gefs_', '')
        if alt_obs_col in df.columns:
            print(f"Using '{alt_obs_col}' instead of '{obs_col}'")
            obs_col = alt_obs_col
        else:
            print(f"Warning: Neither '{obs_col}' nor '{alt_obs_col}' found in columns")
            # Try the forecast column name from target config
            forecast_col = target_config['forecast_col']
            if forecast_col in df.columns:
                print(f"Using '{forecast_col}' instead")
                obs_col = forecast_col
            else:
                print(f"Error: Cannot find suitable observation column")
                return df, {'error': 'observation column not found'}
    
    if urma_col not in df.columns:
        # Try to find the column without the 'urma_' prefix
        alt_urma_col = urma_col.replace('urma_', '')
        if alt_urma_col in df.columns:
            print(f"Using '{alt_urma_col}' instead of '{urma_col}'")
            urma_col = alt_urma_col
        else:
            print(f"Warning: Neither '{urma_col}' nor '{alt_urma_col}' found in columns")
            # Look for URMA field name
            urma_field = target_config['urma_field']
            if urma_field in df.columns:
                print(f"Using '{urma_field}' instead")
                urma_col = urma_field
            else:
                print(f"Error: Cannot find suitable URMA column")
                return df, {'error': 'URMA column not found'}
    
    print(f"Applying QC filter between '{obs_col}' and '{urma_col}'")
    
    df_work = df.copy()
    df_work['urma_obs_deviation'] = abs(df_work[urma_col] - df_work[obs_col])
    qc_mask = df_work['urma_obs_deviation'] <= QC_THRESHOLD
    df_qc = df_work[qc_mask].copy()
    df_qc = df_qc.drop('urma_obs_deviation', axis=1)

    qc_stats = {
        'total_records': len(df_work),
        'qc_passed': qc_mask.sum(),
        'qc_pass_rate': qc_mask.sum() / len(df_work) if len(df_work) > 0 else 0,
        'threshold_used': QC_THRESHOLD
    }

    return df_qc, qc_stats

# =============================================================================
# MACHINE LEARNING FUNCTIONS
# =============================================================================

def prepare_ml_data(time_matched_qc, target_config):
    """Prepare data for ML training with advanced feature engineering."""
    print(f"Available columns for ML preparation: {list(time_matched_qc.columns)}")
    
    # Create feature and target datasets with flexible column naming
    target_col = target_config['urma_obs_col']  # Use URMA observations as target
    nbm_col = 'nbm_' + target_config['forecast_col']
    gefs_col = 'gefs_' + target_config['forecast_col']
    
    # Check and fix column names for target (use URMA observations)
    if target_col not in time_matched_qc.columns:
        # Try alternative observation column names
        alt_targets = [
            target_config['gefs_obs_col'],  # Fallback to config specified
            target_config['obs_col'],  # Base obs column
            target_config['forecast_col']  # Forecast column as last resort
        ]
        
        for alt_target in alt_targets:
            if alt_target in time_matched_qc.columns:
                target_col = alt_target
                print(f"Using alternative target column: {target_col}")
                break
        else:
            print(f"Error: Cannot find target column. Tried: {[target_config['urma_obs_col']] + alt_targets}")
            return None, None, None
    
    if nbm_col not in time_matched_qc.columns:
        alt_nbm = target_config['nbm_col']
        if alt_nbm in time_matched_qc.columns:
            nbm_col = alt_nbm
        else:
            print(f"Warning: NBM column '{nbm_col}' not found, trying '{alt_nbm}'")
            if alt_nbm not in time_matched_qc.columns:
                print(f"NBM column not available")
                nbm_col = None
    
    if gefs_col not in time_matched_qc.columns:
        alt_gefs = target_config['forecast_col']
        if alt_gefs in time_matched_qc.columns:
            gefs_col = alt_gefs
        else:
            print(f"Warning: GEFS column '{gefs_col}' not found, trying '{alt_gefs}'")
            gefs_col = None
    
    print(f"Using columns - Target: '{target_col}', NBM: '{nbm_col}', GEFS: '{gefs_col}'")
    
    # Reset index and filter for complete data
    df = time_matched_qc.reset_index().set_index(['valid_datetime', 'sid'])
    required_cols = [col for col in [target_col, nbm_col, gefs_col] if col is not None]
    df = df.dropna(subset=required_cols)
    
    # Feature Engineering
    print("Creating engineered features...")
    
    # 1. Forecast differences and ratios (only if NBM is included as predictor)
    if INCLUDE_NBM_AS_PREDICTOR and gefs_col and nbm_col and gefs_col in df.columns and nbm_col in df.columns:
        df['gefs_nbm_diff'] = df[gefs_col] - df[nbm_col]
        df['gefs_nbm_ratio'] = df[gefs_col] / (df[nbm_col] + 1e-6)  # Avoid division by zero
        print("  - Added GEFS-NBM difference and ratio features")
    
    # 2. Ensemble features (only if NBM is included as predictor)
    if INCLUDE_NBM_AS_PREDICTOR and gefs_col in df.columns and nbm_col in df.columns:
        df['ensemble_mean'] = (df[gefs_col] + df[nbm_col]) / 2
        df['ensemble_weighted'] = 0.3 * df[gefs_col] + 0.7 * df[nbm_col]  # NBM weighted higher
        print("  - Added ensemble features")
    
    # 3. Time-based features
    df = df.reset_index()
    df['hour'] = pd.to_datetime(df['valid_datetime']).dt.hour
    df['day_of_year'] = pd.to_datetime(df['valid_datetime']).dt.dayofyear
    df['month'] = pd.to_datetime(df['valid_datetime']).dt.month
    df['season'] = ((pd.to_datetime(df['valid_datetime']).dt.month % 12 + 3) // 3).map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'})
    print("  - Added time-based features")
    
    # 4. Station-based features (one-hot encoding)
    station_dummies = pd.get_dummies(df['sid'], prefix='station')
    df = pd.concat([df, station_dummies], axis=1)
    print("  - Added station dummy variables")
    
    # 5. Seasonal interaction terms
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    print("  - Added seasonal dummy variables")
    
    # 6. NBM bias correction features (only if NBM is included as predictor)
    if INCLUDE_NBM_AS_PREDICTOR and 'urma_' + target_config['obs_col'] in df.columns:
        urma_obs_col = 'urma_' + target_config['obs_col']
        if nbm_col in df.columns:
            df['nbm_bias'] = df[nbm_col] - df[urma_obs_col]
        if gefs_col in df.columns:
            df['gefs_bias'] = df[gefs_col] - df[urma_obs_col]
        print("  - Added bias correction features")
    
    # Set index back
    df = df.set_index(['valid_datetime', 'sid'])
    
    # Select features based on configuration - INCLUDE ALL GEFS ATMOSPHERIC VARIABLES
    if INCLUDE_NBM_AS_PREDICTOR:
        # Include all features except observation columns and non-numeric
        feature_cols = [col for col in df.columns if 'obs' not in col.lower() and col != 'season']
        print(f"  - Including NBM forecasts as predictors")
    else:
        # Include ALL GEFS atmospheric variables plus engineered features
        gefs_atmos_cols = [col for col in df.columns if col.startswith('gefs_') and 'obs' not in col.lower()]
        engineered_cols = [col for col in df.columns if any(term in col for term in ['hour', 'day_of_year', 'month', 'station_', 'season_'])]
        feature_cols = gefs_atmos_cols + engineered_cols
        print(f"  - Including ALL {len(gefs_atmos_cols)} GEFS atmospheric variables as features")
        print(f"  - Plus {len(engineered_cols)} engineered features")
        print(f"  - Excluding NBM forecasts as predictors (baseline comparison only)")
    
    # Check if target column exists
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in final dataset")
        print(f"Available columns: {list(df.columns)}")
        return None, None, None
    
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    print(f"Created {X.shape[1]} features from {len(feature_cols)} candidates")
    print(f"Target variable: {target_col} (shape: {y.shape})")
    return X, y, df

def apply_feature_selection(X, y, splits, n_features=20):
    """Apply feature selection to reduce overfitting and improve performance."""
    if not USE_FEATURE_SELECTION:
        return X, splits, list(X.columns)
    
    print(f"Applying feature selection (selecting top {n_features} features)...")
    print(f"Starting with {X.shape[1]} total features")
    
    # Clean training data for feature selection
    train_mask = ~(splits['X_train'].isnull().any(axis=1) | splits['y_train'].isnull())
    X_train_clean = splits['X_train'][train_mask]
    y_train_clean = splits['y_train'][train_mask]
    
    # Identify GEFS atmospheric variables (these are the most important for weather forecasting)
    gefs_atmos_features = [col for col in X.columns if col.startswith('gefs_') and 'obs' not in col.lower()]
    other_features = [col for col in X.columns if col not in gefs_atmos_features]
    
    print(f"Found {len(gefs_atmos_features)} GEFS atmospheric variables")
    print(f"Found {len(other_features)} other engineered features")
    
    # Method 1: Statistical feature selection (F-test) - be generous with GEFS atmospheric vars
    f_selector = SelectKBest(f_regression, k=min(n_features * 3, X_train_clean.shape[1]))
    f_selector.fit(X_train_clean, y_train_clean)
    f_selected = f_selector.get_support()
    f_scores = f_selector.scores_
    
    # Method 2: Recursive Feature Elimination with Random Forest
    rf_selector = RFE(
        RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1),
        n_features_to_select=min(n_features * 2, X_train_clean.shape[1])
    )
    rf_selector.fit(X_train_clean, y_train_clean)
    rf_selected = rf_selector.get_support()
    
    # Combine both methods (features selected by either method)
    combined_selected = f_selected | rf_selected
    candidate_features = X.columns[combined_selected].tolist()
    
    print(f"F-test selected: {f_selected.sum()} features")
    print(f"RFE selected: {rf_selected.sum()} features") 
    print(f"Combined candidates: {len(candidate_features)} features")
    
    # ALWAYS include high-scoring GEFS atmospheric variables (they're crucial for weather prediction)
    gefs_scores = [(col, f_scores[i]) for i, col in enumerate(X.columns) if col in gefs_atmos_features]
    gefs_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Add top GEFS atmospheric variables that weren't already selected
    top_gefs_features = [col for col, score in gefs_scores[:max(10, n_features//2)]]
    for feature in top_gefs_features:
        if feature not in candidate_features:
            candidate_features.append(feature)
            print(f"Added important GEFS atmospheric variable: {feature}")
    
    # Ensure we have key forecasting features
    if INCLUDE_NBM_AS_PREDICTOR:
        key_features = [col for col in X.columns if any(term in col.lower() for term in ['nbm', 'ensemble'])]
    else:
        key_features = []  # Already handled GEFS features above
    
    # Add key features that weren't already selected
    for feature in key_features:
        if feature not in candidate_features:
            candidate_features.append(feature)
            print(f"Added key feature: {feature}")
    
    # Sort by F-test scores (descending) and take top n_features
    f_scores_dict = dict(zip(X.columns, f_scores))
    candidate_features = sorted(candidate_features, key=lambda x: f_scores_dict.get(x, 0), reverse=True)
    
    # Select final features
    selected_features = candidate_features[:n_features]
    
    print(f"Final selected {len(selected_features)} features:")
    for i, feature in enumerate(selected_features, 1):
        score = f_scores_dict.get(feature, 0)
        print(f"  {i:2d}. {feature:<25} (F-score: {score:.2f})")
    
    # Apply selection to all data
    X_selected = X[selected_features]
    splits_selected = {
        'X_train': splits['X_train'][selected_features],
        'X_val': splits['X_val'][selected_features],
        'X_test': splits['X_test'][selected_features],
        'y_train': splits['y_train'],
        'y_val': splits['y_val'],
        'y_test': splits['y_test']
    }
    
    return X_selected, splits_selected, selected_features

def create_time_splits(X, y, test_size=0.2, val_size=0.1):
    """Create time-based train/validation/test splits."""
    unique_dates = X.index.get_level_values('valid_datetime').unique().sort_values()
    n_dates = len(unique_dates)
    
    test_start_idx = int(n_dates * (1 - test_size))
    val_start_idx = int((n_dates - int(n_dates * test_size)) * (1 - val_size))
    
    train_dates = unique_dates[:val_start_idx]
    val_dates = unique_dates[val_start_idx:test_start_idx]
    test_dates = unique_dates[test_start_idx:]
    
    train_mask = X.index.get_level_values('valid_datetime').isin(train_dates)
    val_mask = X.index.get_level_values('valid_datetime').isin(val_dates)
    test_mask = X.index.get_level_values('valid_datetime').isin(test_dates)
    
    return {
        'X_train': X[train_mask], 'y_train': y[train_mask],
        'X_val': X[val_mask], 'y_val': y[val_mask],
        'X_test': X[test_mask], 'y_test': y[test_mask]
    }

def tune_hyperparameters(splits, use_tuning=True):
    """Tune hyperparameters using validation set to prevent overfitting."""
    if not use_tuning:
        return {}
    
    from sklearn.model_selection import GridSearchCV
    
    # Clean training data
    train_mask = ~(splits['X_train'].isnull().any(axis=1) | splits['y_train'].isnull())
    X_train_clean = splits['X_train'][train_mask]
    y_train_clean = splits['y_train'][train_mask]
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [30, 50, 70],
        'max_depth': [8, 10, 12, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 5, 8],
        'max_features': ['sqrt', 'log2']
    }
    
    # Create base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True, oob_score=True)
    
    # Perform grid search with cross-validation
    print("Tuning hyperparameters...")
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train_clean, y_train_clean)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.3f}")
    
    return grid_search.best_params_

def train_model_with_validation(splits, use_tuning=False):
    """Train Random Forest model with validation monitoring to prevent overfitting."""
    # Clean data
    train_mask = ~(splits['X_train'].isnull().any(axis=1) | splits['y_train'].isnull())
    X_train_clean = splits['X_train'][train_mask]
    y_train_clean = splits['y_train'][train_mask]
    
    val_mask = ~(splits['X_val'].isnull().any(axis=1) | splits['y_val'].isnull())
    X_val_clean = splits['X_val'][val_mask]
    y_val_clean = splits['y_val'][val_mask]
    
    # Test different n_estimators to find optimal stopping point
    n_estimators_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    val_scores = []
    
    print("Finding optimal number of estimators...")
    for n_est in n_estimators_range:
        model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_clean, y_train_clean)
        val_pred = model.predict(X_val_clean)
        val_score = mean_squared_error(y_val_clean, val_pred)
        val_scores.append(val_score)
        print(f"n_estimators={n_est}, val_mse={val_score:.3f}")
    
    # Find optimal n_estimators
    optimal_idx = np.argmin(val_scores)
    optimal_n_estimators = n_estimators_range[optimal_idx]
    print(f"Optimal n_estimators: {optimal_n_estimators}")
    
    # Train final model with optimal parameters
    if use_tuning:
        best_params = tune_hyperparameters(splits, use_tuning=True)
        best_params['n_estimators'] = optimal_n_estimators
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1, bootstrap=True, oob_score=True)
    else:
        model = RandomForestRegressor(
            n_estimators=optimal_n_estimators,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train_clean, y_train_clean)
    
    print(f"Final model trained on {len(y_train_clean)} samples")
    print(f"Out-of-bag score: {model.oob_score_:.3f}")
    return model

def train_model(splits, use_tuning=False):
    """Train Random Forest model with regularization to prevent overfitting."""
    # Clean data
    train_mask = ~(splits['X_train'].isnull().any(axis=1) | splits['y_train'].isnull())
    X_train_clean = splits['X_train'][train_mask]
    y_train_clean = splits['y_train'][train_mask]
    
    # Get hyperparameters
    if use_tuning:
        best_params = tune_hyperparameters(splits, use_tuning=True)
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1, bootstrap=True, oob_score=True)
    else:
        # Train model with regularization parameters to prevent overfitting
        model = RandomForestRegressor(
            n_estimators=50,        # Reduced from 100 to prevent overfitting
            max_depth=10,           # Limit tree depth
            min_samples_split=10,   # Require more samples to split
            min_samples_leaf=5,     # Require more samples in leaf nodes
            max_features='sqrt',    # Use subset of features per tree
            bootstrap=True,         # Enable bootstrapping
            oob_score=True,         # Out-of-bag score for monitoring
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train_clean, y_train_clean)
    
    print(f"Model trained on {len(y_train_clean)} samples")
    print(f"Out-of-bag score: {model.oob_score_:.3f}")
    return model

def train_ensemble_models(splits, use_tuning=False):
    """Train multiple models and create an ensemble."""
    # Clean data
    train_mask = ~(splits['X_train'].isnull().any(axis=1) | splits['y_train'].isnull())
    X_train_clean = splits['X_train'][train_mask]
    y_train_clean = splits['y_train'][train_mask]
    
    val_mask = ~(splits['X_val'].isnull().any(axis=1) | splits['y_val'].isnull())
    X_val_clean = splits['X_val'][val_mask]
    y_val_clean = splits['y_val'][val_mask]
    
    print("Training ensemble of models...")
    
    # Define individual models with regularization
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=50, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=50, max_depth=6, learning_rate=0.1,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000)
    }
    
    # Train individual models and evaluate on validation set
    trained_models = {}
    val_scores = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_clean, y_train_clean)
        
        # Evaluate on validation set
        val_pred = model.predict(X_val_clean)
        val_score = mean_squared_error(y_val_clean, val_pred)
        val_scores[name] = val_score
        trained_models[name] = model
        
        print(f"  {name} validation MSE: {val_score:.3f}")
    
    # Create weighted ensemble based on validation performance
    # Invert scores so better models get higher weights
    inv_scores = {name: 1.0 / score for name, score in val_scores.items()}
    total_inv_score = sum(inv_scores.values())
    weights = {name: score / total_inv_score for name, score in inv_scores.items()}
    
    print(f"\nEnsemble weights: {weights}")
    
    # Create voting ensemble
    estimators = [(name, model) for name, model in trained_models.items()]
    ensemble = VotingRegressor(estimators=estimators, weights=list(weights.values()))
    ensemble.fit(X_train_clean, y_train_clean)
    
    print(f"Ensemble trained on {len(y_train_clean)} samples")
    return ensemble, trained_models

def evaluate_model(model, splits, df, target_config):
    """Evaluate model performance on all splits."""
    results = {}
    predictions = {}
    
    nbm_col = 'nbm_' + target_config['forecast_col']
    
    for split_name in ['train', 'val', 'test']:
        X = splits[f'X_{split_name}']
        y = splits[f'y_{split_name}']
        
        # Clean data for prediction
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(y_clean) == 0:
            continue
            
        # ML predictions
        y_pred = model.predict(X_clean)
        
        # NBM baseline predictions
        nbm_pred = df.loc[y_clean.index, nbm_col].values
        
        # Calculate metrics
        ml_metrics = calculate_metrics(y_clean, y_pred, f'ML_{split_name}')
        nbm_metrics = calculate_metrics(y_clean, nbm_pred, f'NBM_{split_name}')
        
        results[f'ML_{split_name}'] = ml_metrics
        results[f'NBM_{split_name}'] = nbm_metrics
        
        predictions[split_name] = {
            'y_true': y_clean,
            'y_ml': y_pred,
            'y_nbm': nbm_pred
        }
    
    return results, predictions

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate performance metrics."""
    return {
        'model': model_name,
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'bias': np.mean(y_pred - y_true),
        'n_samples': len(y_true)
    }

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_scatter_plot(results, predictions, target_config):
    """Create comprehensive scatter plot for all splits."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Model Performance: {target_config["description"].title()}', fontsize=16)
    
    splits = ['train', 'val', 'test']
    colors = ['blue', 'green', 'red']
    
    for i, split in enumerate(splits):
        if split not in predictions:
            continue
            
        pred_data = predictions[split]
        y_true = pred_data['y_true']
        y_ml = pred_data['y_ml']
        y_nbm = pred_data['y_nbm']
        
        # ML performance (top row)
        ax_ml = axes[0, i]
        ax_ml.scatter(y_true, y_ml, alpha=0.6, color=colors[i], s=20)
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_ml.min()), max(y_true.max(), y_ml.max())
        ax_ml.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
        
        # Metrics text
        ml_r2 = results[f'ML_{split}']['r2']
        ml_mae = results[f'ML_{split}']['mae']
        ml_rmse = results[f'ML_{split}']['rmse']
        
        metrics_text = f'R² = {ml_r2:.3f}\nMAE = {ml_mae:.2f}°C\nRMSE = {ml_rmse:.2f}°C'
        ax_ml.text(0.05, 0.95, metrics_text, transform=ax_ml.transAxes, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_ml.set_title(f'ML Model - {split.title()}')
        ax_ml.set_xlabel('Observed (°C)')
        ax_ml.set_ylabel('Predicted (°C)')
        ax_ml.grid(True, alpha=0.3)
        
        # NBM performance (bottom row)
        ax_nbm = axes[1, i]
        ax_nbm.scatter(y_true, y_nbm, alpha=0.6, color=colors[i], s=20)
        ax_nbm.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
        
        nbm_r2 = results[f'NBM_{split}']['r2']
        nbm_mae = results[f'NBM_{split}']['mae']
        nbm_rmse = results[f'NBM_{split}']['rmse']
        
        metrics_text = f'R² = {nbm_r2:.3f}\nMAE = {nbm_mae:.2f}°C\nRMSE = {nbm_rmse:.2f}°C'
        ax_nbm.text(0.05, 0.95, metrics_text, transform=ax_nbm.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_nbm.set_title(f'NBM Baseline - {split.title()}')
        ax_nbm.set_xlabel('Observed (°C)')
        ax_nbm.set_ylabel('Predicted (°C)')
        ax_nbm.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path('./plots')
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f'model_evaluation_{target_config["target_type"]}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to: {plot_path}")

def create_feature_importance_plot(model, feature_names, target_config, all_features=None):
    """Create clean feature importance plot showing all features."""
    # Extract feature importances
    importances = None
    rf_estimator = None
    
    if hasattr(model, 'feature_importances_'):
        # Single model (e.g., RandomForest)
        importances = model.feature_importances_
        print(f"Using feature importances from single model: {type(model).__name__}")
    elif hasattr(model, 'estimators_'):
        # Ensemble model - get average importance from RandomForest if available
        print(f"Ensemble model detected: {type(model).__name__}")
        
        # Handle VotingRegressor
        if hasattr(model, 'named_estimators_'):
            # VotingRegressor has named_estimators_
            print(f"Available estimators: {list(model.named_estimators_.keys())}")
            for name, estimator in model.named_estimators_.items():
                print(f"  - {name}: {type(estimator).__name__}, has feature_importances_: {hasattr(estimator, 'feature_importances_')}")
                if hasattr(estimator, 'feature_importances_'):
                    rf_estimator = estimator
                    importances = estimator.feature_importances_
                    print(f"Using feature importances from {name}")
                    break
        else:
            # Other ensemble types
            for estimator in model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    rf_estimator = estimator
                    importances = estimator.feature_importances_
                    break
                    
        if importances is None:
            print("Warning: No feature importances available for this ensemble model")
            return
    else:
        print("Warning: Model does not support feature importance extraction")
        return
    
    # Debug: Print lengths to identify mismatch
    print(f"Debug: feature_names length: {len(feature_names)}")
    print(f"Debug: importances length: {len(importances)}")
    print(f"Debug: feature_names type: {type(feature_names)}")
    print(f"Debug: importances type: {type(importances)}")
    
    # Ensure lengths match
    if len(feature_names) != len(importances):
        print(f"Warning: Length mismatch between feature_names ({len(feature_names)}) and importances ({len(importances)})")
        # Take the minimum length to avoid errors
        min_length = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_length]
        importances = importances[:min_length]
        print(f"Truncated both to length: {min_length}")
    
    # Create comprehensive feature analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(f'Feature Analysis: {target_config["description"].title()}', fontsize=16)
    
    # Left plot: Used features with importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Plot all used features
    bars = ax1.barh(range(len(importance_df)), importance_df['importance'], color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(importance_df)))
    ax1.set_yticklabels(importance_df['feature'], fontsize=9)
    ax1.set_xlabel('Feature Importance', fontsize=12)
    ax1.set_title(f'Used Features (n={len(feature_names)})', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # Right plot: All features showing what was selected vs not selected
    if all_features is not None:
        used_features = set(feature_names)
        unused_features = [f for f in all_features if f not in used_features]
        
        # Create feature categories
        feature_categories = []
        colors = []
        for feature in all_features:
            if feature in used_features:
                feature_categories.append(f"✓ {feature}")
                colors.append('darkgreen')
            else:
                feature_categories.append(f"✗ {feature}")
                colors.append('darkred')
        
        # Plot all features
        y_pos = range(len(feature_categories))
        ax2.barh(y_pos, [1] * len(feature_categories), color=colors, alpha=0.6)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(feature_categories, fontsize=8)
        ax2.set_xlabel('Feature Status', fontsize=12)
        ax2.set_title(f'All Features (✓ Used: {len(used_features)}, ✗ Unused: {len(unused_features)})', fontsize=12)
        ax2.set_xlim(0, 1.2)
        
        # Remove x-axis ticks since they're not meaningful
        ax2.set_xticks([])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='darkgreen', alpha=0.6, label='Used'),
                          Patch(facecolor='darkred', alpha=0.6, label='Unused')]
        ax2.legend(handles=legend_elements, loc='lower right')
    else:
        ax2.text(0.5, 0.5, 'All features list not provided', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('All Features (Not Available)', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path('./plots')
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f'feature_analysis_{target_config["target_type"]}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature analysis plot saved to: {plot_path}")
    
    # Print feature summary
    print(f"\nFeature Importance Summary:")
    print(f"Top 5 most important features:")
    for i, (_, row) in enumerate(importance_df.tail(5).iterrows(), 1):
        print(f"  {i}. {row['feature']:<25} {row['importance']:.4f}")
    
    if all_features:
        print(f"\nUnused features that might be valuable:")
        unused = [f for f in all_features if f not in feature_names]
        key_unused = [f for f in unused if any(term in f.lower() for term in ['nbm', 'gefs', 'ensemble', 'bias'])]
        for feature in key_unused[:5]:
            print(f"  - {feature}")

def create_residuals_plot(predictions, target_config):
    """Create clean residuals analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Residuals Analysis: {target_config["description"].title()}', fontsize=16)
    
    # Use test set for residuals analysis
    if 'test' not in predictions:
        print("Warning: No test predictions available for residuals analysis")
        return
    
    pred_data = predictions['test']
    y_true = pred_data['y_true']
    y_ml = pred_data['y_ml']
    y_nbm = pred_data['y_nbm']
    
    ml_residuals = y_ml - y_true
    nbm_residuals = y_nbm - y_true
    
    # 1. Residuals vs Predicted (ML)
    ax1 = axes[0, 0]
    ax1.scatter(y_ml, ml_residuals, alpha=0.6, s=20, color='blue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax1.set_xlabel('ML Predicted (°C)')
    ax1.set_ylabel('Residuals (°C)')
    ax1.set_title('ML Model: Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted (NBM)
    ax2 = axes[0, 1]
    ax2.scatter(y_nbm, nbm_residuals, alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel('NBM Predicted (°C)')
    ax2.set_ylabel('Residuals (°C)')
    ax2.set_title('NBM Model: Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram of residuals (ML)
    ax3 = axes[1, 0]
    ax3.hist(ml_residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax3.set_xlabel('ML Residuals (°C)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('ML Model: Residuals Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics text
    ml_mean = np.mean(ml_residuals)
    ml_std = np.std(ml_residuals)
    ax3.text(0.05, 0.95, f'Mean: {ml_mean:.3f}°C\nStd: {ml_std:.3f}°C', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Histogram of residuals (NBM)
    ax4 = axes[1, 1]
    ax4.hist(nbm_residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_xlabel('NBM Residuals (°C)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('NBM Model: Residuals Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    nbm_mean = np.mean(nbm_residuals)
    nbm_std = np.std(nbm_residuals)
    ax4.text(0.05, 0.95, f'Mean: {nbm_mean:.3f}°C\nStd: {nbm_std:.3f}°C', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path('./plots')
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f'residuals_analysis_{target_config["target_type"]}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals analysis plot saved to: {plot_path}")

def print_results_summary(results, qc_stats, target_config):
    """Print comprehensive results summary."""
    print(f"\n=== RESULTS SUMMARY: {target_config['description'].upper()} ===")
    
    # Handle QC statistics (check if it's an error or valid stats)
    if isinstance(qc_stats, dict) and 'error' in qc_stats:
        print(f"QC Statistics:")
        print(f"  - Error: {qc_stats['error']}")
        print(f"  - QC was not properly applied due to column issues")
    elif isinstance(qc_stats, dict) and 'total_records' in qc_stats:
        print(f"QC Statistics:")
        print(f"  - Total records: {qc_stats['total_records']:,}")
        print(f"  - Records passing QC: {qc_stats['qc_passed']:,}")
        print(f"  - QC pass rate: {qc_stats['qc_pass_rate']*100:.1f}%")
    else:
        print(f"QC Statistics:")
        print(f"  - QC stats unavailable or malformed: {qc_stats}")
    
    print(f"\nModel Performance Comparison:")
    print(f"{'Split':<8} {'Model':<12} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Samples':<8}")
    print("-" * 60)
    
    for split in ['train', 'val', 'test']:
        for model_type in ['ML', 'NBM']:
            key = f'{model_type}_{split}'
            if key in results:
                r = results[key]
                print(f"{split:<8} {model_type:<12} {r['mae']:<8.2f} {r['rmse']:<8.2f} "
                      f"{r['r2']:<8.3f} {r['n_samples']:<8,}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=== GEFS ML TRAINING PIPELINE (SIMPLIFIED) ===")
    
    # Get target configuration
    target_config = get_target_config()
    print(f"Target: {target_config['description']} ({target_config['target_type']})")
    print(f"NBM as predictor: {'Yes' if INCLUDE_NBM_AS_PREDICTOR else 'No (baseline only)'}")
    
    # Load data
    print("\n=== LOADING DATA ===")
    forecast, nbm, urma = load_combined_data()
    
    # Prepare data subsets - include ALL GEFS variables (atmospheric AND observations)
    print("Preparing forecast data...")
    
    # Keep ALL forecast columns (including observations for QC)
    forecast_cols = list(forecast.columns)
    obs_columns = ['tmax_obs', 'tmin_obs']
    gefs_atmos_cols = [col for col in forecast_cols if col not in obs_columns]
    gefs_obs_cols = [col for col in forecast_cols if col in obs_columns]
    
    print(f"Using {len(gefs_atmos_cols)} GEFS atmospheric variables: {gefs_atmos_cols}")
    print(f"Using {len(gefs_obs_cols)} GEFS observation variables: {gefs_obs_cols}")
    
    # For NBM, keep the target variable and observations
    nbm_cols = [target_config['forecast_col'], target_config['obs_col']]
    
    forecast_subset = forecast[forecast_cols].dropna().sort_index()
    nbm_subset = nbm[nbm_cols].dropna().sort_index()
    urma_subset = urma[[target_config['urma_field']]].dropna()
    
    # Standardize column names - split forecast data properly
    
    # Split forecast data into atmospheric variables and observations
    forecast_atmos = forecast_subset[gefs_atmos_cols]
    forecast_obs = forecast_subset[gefs_obs_cols] if gefs_obs_cols else pd.DataFrame()
    
    # Add prefixes
    forecast_atmos.columns = ['gefs_' + col for col in forecast_atmos.columns]
    forecast_obs.columns = ['gefs_' + col for col in forecast_obs.columns] if not forecast_obs.empty else []
    
    # Recombine forecast data
    forecast_subset = pd.concat([forecast_atmos, forecast_obs], axis=1)
    
    nbm_subset.columns = ['nbm_' + col for col in nbm_subset.columns]
    
    # Standardize URMA column names
    urma_subset = urma_subset.reset_index().rename(
        columns={target_config['urma_field']: target_config['obs_col'],
                 'valid_time': 'valid_datetime', 'station_id': 'sid'}
    ).set_index(['valid_datetime', 'sid']).sort_index()
    
    # Add prefixes and convert units (K to C)
    urma_subset.columns = ['urma_' + col for col in urma_subset.columns]
    urma_subset[target_config['urma_obs_col']] -= 273.15  # K to C
    
    # Create time-matched dataset
    print("\n=== CREATING TIME-MATCHED DATASET ===")
    time_matched_df = create_time_matched_dataset(forecast_subset, nbm_subset, urma_subset)
    
    # Apply quality control
    print("\n=== APPLYING QUALITY CONTROL ===")
    time_matched_qc, qc_stats = apply_qc_filter(time_matched_df, target_config)
    
    # Prepare ML data
    print("\n=== PREPARING ML DATA ===")
    X, y, df = prepare_ml_data(time_matched_qc, target_config)
    
    # Create splits
    print("\n=== CREATING TIME-BASED SPLITS ===")
    splits = create_time_splits(X, y)
    
    # Apply feature selection
    print("\n=== FEATURE SELECTION ===")
    original_features = list(X.columns)  # Store original feature list
    X, splits, selected_features = apply_feature_selection(X, y, splits, n_features=35)  # Increased from 20 to 35
    
    # Train model
    print("\n=== TRAINING MODEL ===")
    if USE_ENSEMBLE_MODELS:
        model, individual_models = train_ensemble_models(splits, use_tuning=USE_HYPERPARAMETER_TUNING)
    elif USE_VALIDATION_BASED_TRAINING:
        model = train_model_with_validation(splits, use_tuning=USE_HYPERPARAMETER_TUNING)
    else:
        model = train_model(splits, use_tuning=USE_HYPERPARAMETER_TUNING)
    
    # Evaluate model
    print("\n=== EVALUATING MODEL ===")
    results, predictions = evaluate_model(model, splits, df, target_config)
    
    # Create visualization
    print("\n=== CREATING VISUALIZATION ===")
    create_scatter_plot(results, predictions, target_config)
    create_feature_importance_plot(model, selected_features, target_config, original_features)
    create_residuals_plot(predictions, target_config)
    
    # Print summary
    print_results_summary(results, qc_stats, target_config)
    
    print("\n=== PIPELINE COMPLETED ===")
    
    return {
        'model': model,
        'results': results,
        'predictions': predictions,
        'qc_stats': qc_stats,
        'target_config': target_config
    }

if __name__ == "__main__":
    results = main()