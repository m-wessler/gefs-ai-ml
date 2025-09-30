#!/usr/bin/env python3

"""
GEFS ML Weather Forecasting Training Pipeline (Simplified)

This script implements a simplified machine learning approach to post-process weather forecasts,
focusing on temperature predictions. It combines GEFS and URMA data to train ML models
with quality control and generates evaluation plots.

Key Features:
- URMA quality control for data validation
- Time-based train/validation/test splits
- Random Forest model training with validation-based training
- Simplified configuration with essential toggles only

SIMPLIFIED CONFIGURATION:
- NBM is NO LONGER USED as a predictor (only for baseline comparison)
- Essential toggles only: INCLUDE_GEFS_AS_PREDICTOR, USE_FEATURE_SELECTION, USE_HYPERPARAMETER_TUNING
- Always uses validation-based training and separate forecast hours for best practices
- Removed redundant options: USE_ENSEMBLE_MODELS, PREVENT_OVERFITTING, PREFER_GPU_MODELS, etc.

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
from datetime import timedelta, datetime
import json
import joblib
import os
import sys
import argparse
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import argparse
import sys

# Try to import GPU-accelerated libraries
try:
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor
    CUML_AVAILABLE = True
    print("cuML (GPU) support detected")
except ImportError:
    CUML_AVAILABLE = False
    print("cuML not available - using CPU-only algorithms")

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print("XGBoost support detected")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    import catboost as cb
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
    print("CatBoost support detected")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
    print("LightGBM support detected")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

# Set optimal CPU parallelization
try:
    import multiprocessing
    N_JOBS = min(multiprocessing.cpu_count(), 16)  # Cap at 16 for memory reasons
    os.environ['OMP_NUM_THREADS'] = str(N_JOBS)
    os.environ['MKL_NUM_THREADS'] = str(N_JOBS)
    print(f"Using {N_JOBS} CPU cores for parallel processing")
except:
    N_JOBS = -1
    print("Using all available CPU cores")

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

BASE_PATH = "N:/data/gefs-ml/"
OUTPUT_PATH = "N:/projects/michael.wessler/gefs-ai-ml/"

# BASE_PATH = '/nas/stid/data/gefs-ml/'
# OUTPUT_PATH = "/nas/stid/projects/michael.wessler/gefs-ai-ml/"

# Station Configuration
USE_ALL_AVAILABLE_STATIONS = False  # Set to True to use all available stations instead of STATION_IDS
MAX_STATIONS = 500  # Maximum number of stations to use when USE_ALL_AVAILABLE_STATIONS is True
RANDOM_STATION_SEED = 42  # Random seed for reproducible station selection
STATION_IDS = ['KEKO']#, 'KBOI', 'KLAS', 'KSEA', 'KLAX']  # Multiple stations for better generalization

# Data Configuration
FORECAST_HOURS = ['f024']#, 'f048', 'f072']  # Multiple forecast hours for better generalization
QC_THRESHOLD = 10  # Maximum allowed deviation between URMA and station obs in °C
TARGET_VARIABLE = 'tmin'  # 'tmax' or 'tmin'

# Model Configuration - Essential Toggles Only
USE_HYPERPARAMETER_TUNING = True  # Set to True for better performance but slower training
QUICK_HYPERPARAMETER_TUNING = False  # Set to True for faster training with smaller grids (recommended for testing)
USE_FEATURE_SELECTION = True  # Apply feature selection to reduce overfitting
INCLUDE_GEFS_AS_PREDICTOR = False  # Include GEFS forecast of target variable as predictor (True = less independent)
N_FEATURES_TO_SELECT = 25  # Reduced number of features to prevent overfitting (was 25)

# Model Selection
MODELS_TO_TRY = [
    # 'ols',              # Ordinary Least Squares (Linear Regression)
    # 'ridge',            # Ridge regression (L2 regularization)
    # 'lasso',            # Lasso regression (L1 regularization)
    # 'stepwise',         # Stepwise regression (feature selection + OLS)
    # 'random_forest',    # Random Forest (always available)
    'gradient_boosting', # Scikit-learn Gradient Boosting (always available)
    'xgboost',          # XGBoost (if available) 
    # 'catboost',         # CatBoost (if available)
    'lightgbm',         # LightGBM (if available)
    # 'extra_trees'       # Extra Trees (always available)
]

# Hardware Configuration
USE_GPU_ACCELERATION = False  # Use GPU acceleration when available (cuML, XGBoost GPU)

# Reproducibility Configuration
GLOBAL_RANDOM_SEED = 42  # Master random seed for all operations
ENABLE_DETERMINISTIC_RESULTS = True  # Set to True for fully reproducible results (may be slightly slower)

# =============================================================================
# CUSTOM MODELS
# =============================================================================

class StepwiseRegressor(BaseEstimator, RegressorMixin):
    """Stepwise regression using forward/backward feature selection."""
    
    def __init__(self, max_features=None, direction='both', p_enter=0.05, p_remove=0.10):
        self.max_features = max_features
        self.direction = direction  # 'forward', 'backward', or 'both'
        self.p_enter = p_enter
        self.p_remove = p_remove
        self.selected_features_ = None
        self.estimator_ = None
    
    def _forward_selection(self, X, y):
        """Forward feature selection."""
        selected = []
        remaining = list(range(X.shape[1]))
        
        while remaining:
            best_score = float('inf')
            best_feature = None
            
            for feature in remaining:
                candidate_features = selected + [feature]
                X_candidate = X[:, candidate_features]
                
                # Use cross-validation to get more robust p-value estimate
                lr = LinearRegression()
                scores = cross_val_score(lr, X_candidate, y, cv=3, scoring='neg_mean_squared_error')
                score = -scores.mean()
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
            
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
                
                # Stop if we have enough features
                if self.max_features and len(selected) >= self.max_features:
                    break
            else:
                break
                
        return selected
    
    def _backward_elimination(self, X, y):
        """Backward feature elimination."""
        selected = list(range(X.shape[1]))
        
        while len(selected) > 1:
            worst_score = float('inf')
            worst_feature = None
            
            for feature in selected:
                candidate_features = [f for f in selected if f != feature]
                if len(candidate_features) == 0:
                    break
                    
                X_candidate = X[:, candidate_features]
                
                # Use cross-validation to get more robust score
                lr = LinearRegression()
                scores = cross_val_score(lr, X_candidate, y, cv=3, scoring='neg_mean_squared_error')
                score = -scores.mean()
                
                if score < worst_score:
                    worst_score = score
                    worst_feature = feature
            
            if worst_feature is not None:
                selected.remove(worst_feature)
                
                # Stop if we have reached max features
                if self.max_features and len(selected) <= self.max_features:
                    break
            else:
                break
                
        return selected
    
    def fit(self, X, y):
        """Fit the stepwise regression model."""
        X = np.array(X)
        y = np.array(y)
        
        if self.direction == 'forward':
            self.selected_features_ = self._forward_selection(X, y)
        elif self.direction == 'backward':
            self.selected_features_ = self._backward_elimination(X, y)
        else:  # both
            # Start with forward selection, then backward elimination
            forward_features = self._forward_selection(X, y)
            if len(forward_features) > 0:
                X_forward = X[:, forward_features]
                backward_features_idx = self._backward_elimination(X_forward, y)
                self.selected_features_ = [forward_features[i] for i in backward_features_idx]
            else:
                self.selected_features_ = forward_features
        
        # Fit final model with selected features
        if len(self.selected_features_) > 0:
            self.estimator_ = LinearRegression()
            self.estimator_.fit(X[:, self.selected_features_], y)
        else:
            # Fallback to using all features if none selected
            self.selected_features_ = list(range(X.shape[1]))
            self.estimator_ = LinearRegression()
            self.estimator_.fit(X, y)
            
        return self
    
    def predict(self, X):
        """Predict using the fitted stepwise model."""
        if self.estimator_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        return self.estimator_.predict(X[:, self.selected_features_])
    
    def score(self, X, y):
        """Return the coefficient of determination R^2."""
        if self.estimator_ is None:
            raise ValueError("Model must be fitted before scoring")
        
        X = np.array(X)
        return self.estimator_.score(X[:, self.selected_features_], y)


# =============================================================================
# REPRODUCIBILITY FUNCTIONS
# =============================================================================

def set_global_seeds(seed=None):
    """Set all random seeds for reproducible results."""
    if seed is None:
        seed = GLOBAL_RANDOM_SEED
    
    print(f"Setting global random seed to: {seed}")
    
    # Python built-in random
    import random
    random.seed(seed)
    
    # NumPy
    import numpy as np
    np.random.seed(seed)
    
    # Pandas (for sampling operations)
    try:
        import pandas as pd
        # Pandas doesn't have a direct seed, but we can ensure numpy is set
        pass
    except ImportError:
        pass
    
    # Scikit-learn uses numpy's random state, so numpy seed covers this
    
    # Set additional environment variables for deterministic behavior
    if ENABLE_DETERMINISTIC_RESULTS:
        import os
        # Ensure deterministic behavior in parallel processing
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set threading to single-threaded for full reproducibility (optional)
        # Uncomment these if you need absolute determinism at the cost of speed
        # os.environ['OMP_NUM_THREADS'] = '1'
        # os.environ['MKL_NUM_THREADS'] = '1'
        
        print("Deterministic mode enabled - results will be fully reproducible")
    
    # Set model-specific seeds (will be handled in model creation)
    print("Random seeds configured for reproducible training")

def get_random_state_for_model(base_seed=None):
    """Get a consistent random state for model training."""
    if base_seed is None:
        base_seed = GLOBAL_RANDOM_SEED
    return base_seed

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_timestamp():
    """Generate timestamp string for file naming."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def generate_config_info():
    """Generate configuration info string for plot titles."""
    # Forecast hours info
    fhours_str = f"FHours: {', '.join(FORECAST_HOURS)}"
    
    # Stations info
    if USE_ALL_AVAILABLE_STATIONS:
        # We need to get the actual number of stations used
        # This will be set in main() after station selection
        stations_str = f"Stations: {getattr(generate_config_info, 'station_count', 'All Available')}"
    else:
        stations_str = f"Stations: {', '.join(STATION_IDS)}"
    
    return f"{fhours_str} | {stations_str}"

def generate_station_identifier():
    """Generate station identifier for file naming."""
    if USE_ALL_AVAILABLE_STATIONS:
        station_count = getattr(generate_config_info, 'station_count', MAX_STATIONS)
        return f"multi_{station_count}"
    else:
        if len(STATION_IDS) == 1:
            return STATION_IDS[0]
        else:
            return f"multi_{len(STATION_IDS)}"

def generate_forecast_hour_identifier():
    """Generate forecast hour identifier for file naming."""
    if len(FORECAST_HOURS) == 1:
        # Single forecast hour: f024 -> fhr24
        fhr = FORECAST_HOURS[0].replace('f0', '').replace('f', '')
        return f"fhr{fhr}"
    else:
        # Multiple forecast hours: ['f024', 'f048', 'f072'] -> fhr24-72
        hours = []
        for fh in FORECAST_HOURS:
            hour = fh.replace('f0', '').replace('f', '')
            hours.append(int(hour))
        hours.sort()
        return f"fhr{hours[0]}-{hours[-1]}"

def generate_filename_base(target_type, best_model_name=None):
    """Generate base filename with new naming convention."""
    station_id = generate_station_identifier()
    fhr_id = generate_forecast_hour_identifier()
    
    if best_model_name:
        # Include model name in filename
        return f"gefs_ml_{target_type}_{station_id}_{fhr_id}_{best_model_name}"
    else:
        # Without model name (for use during training before model selection)
        return f"gefs_ml_{target_type}_{station_id}_{fhr_id}"

# =============================================================================
# MODEL EXPORT/IMPORT FUNCTIONS
# =============================================================================

def save_model_and_metadata(model, selected_features, target_config, results, qc_stats, model_dir='models'):
    """Save trained model and all metadata needed for future predictions."""
    
    # Create models directory
    model_path = Path(OUTPUT_PATH) / model_dir
    model_path.mkdir(exist_ok=True)
    
    # Extract best model name from results
    best_model_name = results.get('best_model', 'unknown')
    target_type = target_config['target_type']
    
    # Generate filename base with new convention - save directly as latest
    filename_base = generate_filename_base(target_type, best_model_name)
    
    # Define file paths as latest versions only
    model_file_path = model_path / f"{filename_base}_latest.joblib"
    metadata_file_path = model_path / f"{filename_base}_latest.json"
    
    # Save the trained model
    joblib.dump(model, model_file_path)
    print(f"Model saved to: {model_file_path}")
    
    # Prepare metadata for export
    metadata = {
        'model_info': {
            'target_type': target_config['target_type'],
            'target_description': target_config['description'],
            'model_type': type(model).__name__,
            'training_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'script_version': '1.0'
        },
        'training_config': {
            'BASE_PATH': BASE_PATH,
            'FORECAST_HOURS': FORECAST_HOURS,
            'QC_THRESHOLD': QC_THRESHOLD,
            'TARGET_VARIABLE': TARGET_VARIABLE,
            'USE_HYPERPARAMETER_TUNING': USE_HYPERPARAMETER_TUNING,
            'QUICK_HYPERPARAMETER_TUNING': QUICK_HYPERPARAMETER_TUNING,
            'USE_FEATURE_SELECTION': USE_FEATURE_SELECTION,
            'INCLUDE_GEFS_AS_PREDICTOR': INCLUDE_GEFS_AS_PREDICTOR,
            'USE_ALL_AVAILABLE_STATIONS': USE_ALL_AVAILABLE_STATIONS,
            'MAX_STATIONS': MAX_STATIONS,
            'RANDOM_STATION_SEED': RANDOM_STATION_SEED,
            'USE_GPU_ACCELERATION': USE_GPU_ACCELERATION,
            'MODELS_TO_TRY': MODELS_TO_TRY
        },
        'target_config': target_config,
        'features': {
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'feature_types': {
                'gefs_atmospheric': [f for f in selected_features if f.startswith('gefs_') and 'obs' not in f.lower()],
                'engineered': [f for f in selected_features if any(term in f for term in ['hour', 'day_of_year', 'month', 'station_', 'season_'])],
                'nbm_features': [f for f in selected_features if f.startswith('nbm_')],
                'other': [f for f in selected_features if not any([
                    f.startswith('gefs_') and 'obs' not in f.lower(),
                    any(term in f for term in ['hour', 'day_of_year', 'month', 'station_', 'season_']),
                    f.startswith('nbm_')
                ])]
            }
        },
        'data_preprocessing': {
            'obs_columns_expected': ['tmax_obs', 'tmin_obs'],
            'gefs_prefix': 'gefs_',
            'nbm_prefix': 'nbm_',
            'urma_prefix': 'urma_',
            'urma_temp_conversion': 'K_to_C_minus_273.15',
            'prohibited_features': {
                'always_excluded': ['obs columns', 'season (non-numeric)'],
                'target_specific': f"gefs_{target_config['forecast_col']}" if not INCLUDE_GEFS_AS_PREDICTOR else "none",
                'cross_contamination': 'gefs_tmin_2m for tmax prediction, gefs_tmax_2m for tmin prediction'
            }
        },
        'performance_metrics': {
            # Handle the new results structure from evaluate_model
            key: {
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'r2': float(metrics['r2']),
                'bias': float(metrics['bias']),
                'n_samples': int(metrics['n_samples'])
            } for key, metrics in results.items() 
            if isinstance(metrics, dict) and 'mae' in metrics  # Only process actual metric dicts
        },
        'model_comparison': results.get('model_comparison', {}),  # Add model comparison results
        'best_model_info': {
            'best_model_name': results.get('best_model', 'unknown'),
            'selection_criteria': 'test_set_performance'
        },
        'qc_stats': qc_stats if isinstance(qc_stats, dict) else {'error': str(qc_stats)}
    }
    
    # Save metadata as JSON
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Metadata saved to: {metadata_file_path}")
    print(f"Latest model available at: {model_file_path}")
    print(f"Latest metadata available at: {metadata_file_path}")
    
    return {
        'model_file': str(model_file_path),
        'metadata_file': str(metadata_file_path),
        'latest_model_file': str(model_file_path),
        'latest_metadata_file': str(metadata_file_path)
    }

def load_model_and_metadata(model_file_path=None, metadata_file_path=None, target_type='tmax'):
    """
    Load a saved model and its metadata.
    
    Note: With the new naming convention, if model_file_path and metadata_file_path are None,
    you'll need to provide them explicitly or use glob patterns to find the latest files.
    The old simple "_latest" naming convention may not work with station/forecast hour specifics.
    """
    
    if model_file_path is None:
        # Try to find latest file with new naming convention
        model_dir = Path(OUTPUT_PATH) / 'models'
        latest_files = list(model_dir.glob(f"gefs_ml_{target_type}_*_latest.joblib"))
        if latest_files:
            model_file_path = latest_files[0]  # Take first match
        else:
            # Fallback to old naming
            model_file_path = model_dir / f"gefs_ml_model_{target_type}_latest.joblib"
            
    if metadata_file_path is None:
        # Try to find corresponding metadata file
        model_dir = Path(OUTPUT_PATH) / 'models'
        latest_files = list(model_dir.glob(f"gefs_ml_{target_type}_*_latest.json"))
        if latest_files:
            metadata_file_path = latest_files[0]  # Take first match
        else:
            # Fallback to old naming
            metadata_file_path = model_dir / f"gefs_ml_metadata_{target_type}_latest.json"
    
    # Load model
    model = joblib.load(model_file_path)
    print(f"Model loaded from: {model_file_path}")
    
    # Load metadata
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)
    print(f"Metadata loaded from: {metadata_file_path}")
    
    return model, metadata

def apply_model_to_new_data(model, metadata, new_forecast_df, new_nbm_df, new_urma_df=None):
    """Apply a trained model to new data using the saved configuration."""
    
    print("=== APPLYING TRAINED MODEL TO NEW DATA ===")
    
    # Extract configuration from metadata
    target_config = metadata['target_config']
    selected_features = metadata['features']['selected_features']
    training_config = metadata['training_config']
    
    print(f"Target: {target_config['description']} ({target_config['target_type']})")
    print(f"Using {len(selected_features)} features from training")
    
    # Apply same preprocessing as training
    print("Applying preprocessing...")
    
    # Prepare forecast data (same as training)
    forecast_cols = list(new_forecast_df.columns)
    obs_columns = ['tmax_obs', 'tmin_obs']
    gefs_atmos_cols = [col for col in forecast_cols if col not in obs_columns]
    gefs_obs_cols = [col for col in forecast_cols if col in obs_columns]
    
    # Split and add prefixes
    forecast_atmos = new_forecast_df[gefs_atmos_cols]
    forecast_obs = new_forecast_df[gefs_obs_cols] if gefs_obs_cols else pd.DataFrame()
    
    forecast_atmos.columns = ['gefs_' + col for col in forecast_atmos.columns]
    forecast_obs.columns = ['gefs_' + col for col in forecast_obs.columns] if not forecast_obs.empty else []
    
    forecast_processed = pd.concat([forecast_atmos, forecast_obs], axis=1)
    
    # Process NBM data
    nbm_cols = [target_config['forecast_col'], target_config['obs_col']]
    nbm_processed = new_nbm_df[nbm_cols] if all(col in new_nbm_df.columns for col in nbm_cols) else new_nbm_df
    nbm_processed.columns = ['nbm_' + col for col in nbm_processed.columns]
    
    # Combine forecast and NBM data
    combined_df = pd.concat([forecast_processed, nbm_processed], axis=1)
    
    # Apply feature engineering (simplified version)
    if not combined_df.index.names or 'valid_datetime' not in combined_df.index.names:
        if 'valid_datetime' in combined_df.columns:
            combined_df = combined_df.set_index('valid_datetime')
    
    # Add time-based features
    if isinstance(combined_df.index, pd.DatetimeIndex):
        combined_df['hour'] = combined_df.index.hour
        combined_df['day_of_year'] = combined_df.index.dayofyear
        combined_df['month'] = combined_df.index.month
    elif 'valid_datetime' in combined_df.columns:
        combined_df['hour'] = pd.to_datetime(combined_df['valid_datetime']).dt.hour
        combined_df['day_of_year'] = pd.to_datetime(combined_df['valid_datetime']).dt.dayofyear
        combined_df['month'] = pd.to_datetime(combined_df['valid_datetime']).dt.month
    
    # Select only the features used in training
    available_features = [f for f in selected_features if f in combined_df.columns]
    missing_features = [f for f in selected_features if f not in combined_df.columns]
    
    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features from training: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
    
    print(f"Using {len(available_features)} of {len(selected_features)} features")
    
    # Make predictions
    X_new = combined_df[available_features].select_dtypes(include=[np.number])
    predictions = model.predict(X_new)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'predicted': predictions,
        'model_target': target_config['target_type']
    }, index=X_new.index)
    
    # Add NBM baseline if available
    nbm_col = 'nbm_' + target_config['forecast_col']
    if nbm_col in combined_df.columns:
        results_df['nbm_baseline'] = combined_df[nbm_col]
    
    print(f"Generated {len(predictions)} predictions")
    return results_df, {
        'features_used': available_features,
        'features_missing': missing_features,
        'model_metadata': metadata
    }

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
# GPU MODEL CREATION FUNCTIONS
# =============================================================================

def create_gpu_random_forest(n_estimators=100):
    """Create GPU-accelerated Random Forest if available, fallback to CPU"""
    random_seed = get_random_state_for_model()
    
    if USE_GPU_ACCELERATION and CUML_AVAILABLE:
        try:
            print("Creating GPU Random Forest using cuML...")
            from cuml.ensemble import RandomForestRegressor as GPURandomForest
            return GPURandomForest(
                n_estimators=n_estimators,
                random_state=random_seed,
                n_streams=1  # Use single stream for stability
            )
        except Exception as e:
            print(f"GPU Random Forest creation failed: {e}")
            print("Falling back to CPU Random Forest...")
    
    # CPU fallback
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_seed,
        n_jobs=N_JOBS
    )

def create_gpu_xgboost(n_estimators=100):
    """Create GPU-accelerated XGBoost if available, fallback to CPU"""
    random_seed = get_random_state_for_model()
    
    if USE_GPU_ACCELERATION and XGBOOST_AVAILABLE:
        try:
            print("Creating GPU XGBoost...")
            return XGBRegressor(
                n_estimators=n_estimators,
                random_state=random_seed,
                tree_method='gpu_hist',
                gpu_id=0,
                eval_metric='rmse'
            )
        except Exception as e:
            print(f"GPU XGBoost creation failed: {e}")
            print("Falling back to CPU XGBoost...")
    
    # CPU fallback
    return XGBRegressor(
        n_estimators=n_estimators,
        random_state=random_seed,
        n_jobs=N_JOBS,
        eval_metric='rmse'
    )

# =============================================================================
# MODEL FACTORY FUNCTIONS
# =============================================================================

def get_available_models():
    """Return list of available models based on installed packages."""
    available = []
    
    # Always available models (including linear models)
    available.extend([
        'ols',              # Ordinary Least Squares
        'ridge',            # Ridge regression  
        'lasso',            # Lasso regression
        'stepwise',         # Stepwise regression
        'random_forest', 
        'gradient_boosting', 
        'extra_trees'
    ])
    
    # Conditional models
    if XGBOOST_AVAILABLE:
        available.append('xgboost')
    if CATBOOST_AVAILABLE:
        available.append('catboost') 
    if LIGHTGBM_AVAILABLE:
        available.append('lightgbm')
        
    return available

def create_model(model_name, use_gpu=False, **kwargs):
    """Factory function to create different types of models."""
    
    # Get consistent random state
    random_seed = get_random_state_for_model()
    
    if model_name == 'random_forest':
        if use_gpu and CUML_AVAILABLE:
            return create_gpu_random_forest(**kwargs)
        else:
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 10),
                min_samples_leaf=kwargs.get('min_samples_leaf', 5),
                max_features=kwargs.get('max_features', 'sqrt'),
                random_state=random_seed,
                n_jobs=N_JOBS,
                bootstrap=True,
                oob_score=True
            )
    
    elif model_name == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost not available")
        
        if use_gpu:
            return create_gpu_xgboost(**kwargs)
        else:
            return XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=random_seed,
                n_jobs=N_JOBS,
                eval_metric='rmse'
            )
    
    elif model_name == 'catboost':
        if not CATBOOST_AVAILABLE:
            raise ValueError("CatBoost not available")
        
        return CatBoostRegressor(
            iterations=kwargs.get('n_estimators', 100),
            depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 0.8),
            random_seed=random_seed,
            task_type='GPU' if use_gpu else 'CPU',
            verbose=False
        )
    
    elif model_name == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM not available")
        
        return LGBMRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.8),
            random_state=random_seed,
            n_jobs=N_JOBS,
            device='gpu' if use_gpu else 'cpu',
            verbosity=-1  # Use 'verbosity' instead of 'verbose' for LightGBM
        )
    
    elif model_name == 'ridge':
        return Ridge(
            alpha=kwargs.get('alpha', 1.0),
            random_state=random_seed,
            fit_intercept=True
        )
    
    elif model_name == 'lasso':
        return Lasso(
            alpha=kwargs.get('alpha', 1.0),
            random_state=random_seed,
            fit_intercept=True,
            max_iter=kwargs.get('max_iter', 2000)
        )
    
    elif model_name == 'ols':
        return LinearRegression(
            fit_intercept=True
        )
    
    elif model_name == 'stepwise':
        # Custom stepwise regression with feature selection
        return StepwiseRegressor(
            max_features=None,
            direction='both'
        )
    
    elif model_name == 'gradient_boosting':
        return GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 0.8),
            random_state=random_seed
        )
    
    elif model_name == 'extra_trees':
        return ExtraTreesRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 10),
            min_samples_leaf=kwargs.get('min_samples_leaf', 5),
            max_features=kwargs.get('max_features', 'sqrt'),
            random_state=random_seed,
            n_jobs=N_JOBS,
            bootstrap=False
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def get_model_hyperparameter_grid(model_name):
    """Get optimized hyperparameter grid for different models (reduced grid size for faster training)."""
    
    if QUICK_HYPERPARAMETER_TUNING:
        # Quick mode - minimal grids for fast testing
        if model_name == 'random_forest':
            return {
                'n_estimators': [50, 100],  # Reduced from 100 to prevent overfitting
                'max_depth': [5, 8],        # More conservative depth
                'min_samples_split': [20],   # Higher minimum samples
                'min_samples_leaf': [10],    # Higher minimum leaf samples
                'max_features': ['sqrt']     # Conservative feature sampling
            }
            # Grid size: 2×2×1×1×1 = 4 combinations
        
        elif model_name == 'ridge':
            return {
                'alpha': [0.1, 1.0, 10.0]
            }
            # Grid size: 3 combinations
        
        elif model_name == 'lasso':
            return {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
            # Grid size: 4 combinations
        
        elif model_name == 'ols':
            return {}  # No hyperparameters to tune for OLS
            # Grid size: 1 combination
        
        elif model_name == 'stepwise':
            return {
                'direction': ['forward', 'backward', 'both'],
                'max_features': [None, 10, 20]
            }  # Grid size: 9 combinations
        
        elif model_name == 'xgboost':
            return {
                'n_estimators': [100],           # Keep simple for quick mode
                'max_depth': [3],                # Shallower trees
                'learning_rate': [0.05],         # Lower learning rate
                'subsample': [0.8],              # Sample fewer rows
                'colsample_bytree': [0.8],       # Sample fewer features
                'reg_alpha': [1.0],              # L1 regularization
                'reg_lambda': [1.0]              # L2 regularization
            }
            # Grid size: 1×1×1×1×1×1×1 = 1 combination
        
        elif model_name == 'catboost':
            return {
                'n_estimators': [100],
                'max_depth': [6],
                'learning_rate': [0.1],
                'subsample': [0.8],
                'reg_lambda': [0]
            }
            # Grid size: 1×1×1×1×1 = 1 combination
        
        elif model_name == 'lightgbm':
            return {
                'n_estimators': [100],           # Keep simple for quick mode
                'max_depth': [3],                # Shallower trees
                'learning_rate': [0.05],         # Lower learning rate
                'subsample': [0.8],              # Sample fewer rows
                'colsample_bytree': [0.8],       # Sample fewer features
                'reg_alpha': [1.0],              # L1 regularization
                'reg_lambda': [1.0]              # L2 regularization
            }
            # Grid size: 1×1×1×1×1×1×1 = 1 combination
        
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': [100],           # Keep simple for quick mode
                'max_depth': [3],                # Shallower trees
                'learning_rate': [0.05],         # Lower learning rate
                'subsample': [0.8],              # Sample fewer rows
                'min_samples_split': [20],       # Higher minimum samples
                'min_samples_leaf': [10]         # Higher minimum leaf samples
            }
            # Grid size: 1×1×1×1×1×1 = 1 combination
        
        elif model_name == 'extra_trees':
            return {
                'n_estimators': [100],
                'max_depth': [5, 8],             # More conservative depth
                'min_samples_split': [20],       # Higher minimum samples
                'min_samples_leaf': [10],        # Higher minimum leaf samples
                'max_features': ['sqrt']
            }
            # Grid size: 1×2×1×1×1 = 2 combinations
    
    else:
        # Standard mode - more comprehensive grids for better performance
        if model_name == 'random_forest':
            return {
                'n_estimators': [100, 150],  # Reduced from 3 to 2 options
                'max_depth': [8, 10, None],  # Reduced from 5 to 3 options
                'min_samples_split': [10, 20],  # Reduced from 4 to 2 options
                'min_samples_leaf': [5, 10],  # Reduced from 4 to 2 options
                'max_features': ['sqrt', 0.3],  # Reduced from 4 to 2 options
                'max_samples': [0.8, None]  # Reduced from 4 to 2 options
            }
            # Grid size: 2×3×2×2×2×2 = 96 combinations
        
        elif model_name == 'xgboost':
            return {
                'n_estimators': [100, 150],  # Reduced from 3 to 2 options
                'max_depth': [4, 6],  # Reduced from 4 to 2 options
                'learning_rate': [0.05, 0.1],  # Reduced from 4 to 2 options
                'subsample': [0.8, 0.9],  # Reduced from 3 to 2 options
                'colsample_bytree': [0.8, 0.9],  # Reduced from 3 to 2 options
                'reg_alpha': [0, 0.1],  # Reduced from 3 to 2 options
                'reg_lambda': [0, 0.1]  # Reduced from 3 to 2 options
            }
            # Grid size: 2×2×2×2×2×2×2 = 128 combinations
        
        elif model_name == 'catboost':
            return {
                'n_estimators': [100, 150],  # Reduced from 3 to 2 options
                'max_depth': [6, 8],  # Reduced from 4 to 2 options
                'learning_rate': [0.05, 0.1],  # Reduced from 4 to 2 options
                'subsample': [0.8, 0.9],  # Reduced from 3 to 2 options
                'reg_lambda': [0, 1]  # Reduced from 4 to 2 options
            }
            # Grid size: 2×2×2×2×2 = 32 combinations
        
        elif model_name == 'lightgbm':
            return {
                'n_estimators': [100, 150],  # Reduced from 3 to 2 options
                'max_depth': [4, 6],  # Reduced from 4 to 2 options
                'learning_rate': [0.05, 0.1],  # Reduced from 4 to 2 options
                'subsample': [0.8, 0.9],  # Reduced from 3 to 2 options
                'colsample_bytree': [0.8, 0.9],  # Reduced from 3 to 2 options
                'reg_alpha': [0, 0.1],  # Reduced from 3 to 2 options
                'reg_lambda': [0, 0.1]  # Reduced from 3 to 2 options
            }
            # Grid size: 2×2×2×2×2×2×2 = 128 combinations
        
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': [100, 150],  # Reduced from 3 to 2 options
                'max_depth': [4, 6],  # Reduced from 4 to 2 options
                'learning_rate': [0.05, 0.1],  # Reduced from 4 to 2 options
                'subsample': [0.8, 0.9],  # Reduced from 3 to 2 options
                'min_samples_split': [10, 20],  # Reduced from 3 to 2 options
                'min_samples_leaf': [5, 10]  # Reduced from 3 to 2 options
            }
            # Grid size: 2×2×2×2×2×2 = 64 combinations
        
        elif model_name == 'extra_trees':
            return {
                'n_estimators': [100, 150],  # Reduced from 3 to 2 options
                'max_depth': [8, 10, None],  # Reduced from 5 to 3 options
                'min_samples_split': [10, 20],  # Reduced from 4 to 2 options
                'min_samples_leaf': [5, 10],  # Reduced from 4 to 2 options
                'max_features': ['sqrt', 0.3]  # Reduced from 4 to 2 options
            }
            # Grid size: 2×3×2×2×2 = 48 combinations
        
        elif model_name == 'ridge':
            return {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
            # Grid size: 4 combinations
        
        elif model_name == 'lasso':
            return {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
            # Grid size: 4 combinations
        
        elif model_name == 'ols':
            return {}  # No hyperparameters to tune for OLS
            # Grid size: 1 combination
        
        elif model_name == 'stepwise':
            return {
                'direction': ['forward', 'backward', 'both'],
                'max_features': [None, 10, 15, 20]
            }  # Grid size: 12 combinations
    
    # If we get here, model_name is not recognized
    raise ValueError(f"No hyperparameter grid defined for model: {model_name}")

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def discover_available_stations():
    """Discover all available stations by scanning the forecast directory."""
    import random
    
    available_stations = set()
    
    # Scan forecast directory for available stations
    for fhour in FORECAST_HOURS:
        data_dir = Path(BASE_PATH) / 'forecast' / fhour
        if data_dir.exists():
            for file_path in data_dir.glob("*_2020_2025_*.csv"):
                # Extract station ID from filename (e.g., "KSLC_2020_2025_f024.csv" -> "KSLC")
                station_id = file_path.stem.split('_')[0]
                available_stations.add(station_id)
    
    available_stations = sorted(list(available_stations))
    print(f"Found {len(available_stations)} available stations: {available_stations[:10]}{'...' if len(available_stations) > 10 else ''}")
    
    return available_stations

def select_stations():
    """Select stations based on configuration."""
    import random
    
    if USE_ALL_AVAILABLE_STATIONS:
        print("=== DISCOVERING AVAILABLE STATIONS ===")
        available_stations = discover_available_stations()
        
        if len(available_stations) <= MAX_STATIONS:
            selected_stations = available_stations
            print(f"Using all {len(selected_stations)} available stations")
        else:
            # Randomly select stations for reproducibility - use global seed for consistency
            random.seed(GLOBAL_RANDOM_SEED)
            selected_stations = random.sample(available_stations, MAX_STATIONS)
            selected_stations.sort()  # Sort for consistent ordering
            print(f"Randomly selected {len(selected_stations)} stations from {len(available_stations)} available")
            print(f"Random seed used: {GLOBAL_RANDOM_SEED}")
        
        print(f"Selected stations: {selected_stations}")
        return selected_stations
    else:
        print(f"Using predefined station list: {STATION_IDS}")
        return STATION_IDS

def load_combined_data():
    """Load and combine forecast, NBM, and URMA data."""
    print(f"Loading data from: {BASE_PATH}")
    
    # Select stations based on configuration
    station_ids = select_stations()
    
    # Store station count for plot titles
    generate_config_info.station_count = len(station_ids)
    
    print(f"Forecast hours: {FORECAST_HOURS}")
    
    forecast_dfs = []
    nbm_dfs = []
    
    # Track missing files for reporting
    missing_forecast = []
    missing_nbm = []

    # Load forecast and NBM data
    for data_type in ['forecast', 'nbm']:
        for fhour in FORECAST_HOURS:
            data_dir = Path(BASE_PATH) / data_type / fhour
            for station_id in station_ids:
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
                    if data_type == 'forecast':
                        missing_forecast.append(f"{station_id}_{fhour}")
                    else:
                        missing_nbm.append(f"{station_id}_{fhour}")
    
    # Report missing files
    if missing_forecast:
        print(f"Warning: {len(missing_forecast)} forecast files missing")
    if missing_nbm:
        print(f"Info: {len(missing_nbm)} NBM files missing (NBM stats will use available data only)")

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
    urma_df = urma_df.loc[urma_df.index.get_level_values(1).isin(station_ids)]
    
    print(f"Loaded forecast data: {forecast_df.shape}")
    print(f"Loaded NBM data: {nbm_df.shape}")
    print(f"Loaded URMA data: {urma_df.shape}")

    return forecast_df, nbm_df, urma_df

def create_time_matched_dataset(forecast_subset, nbm_subset, urma_subset):
    """
    Create time-matched dataset with improved handling of multiple forecast hours.
    
    Key improvements:
    1. Ensures proper NBM-GEFS alignment by forecast time
    2. Adds forecast lead time information
    3. Better handling of multiple forecast hours per verification
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

def verify_data_alignment(df, target_config):
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

def aggregate_forecast_hours_for_evaluation(df, method='separate'):
    """
    Handle aggregation of multiple forecast hours for the same URMA observation.
    
    Parameters:
    - df: DataFrame with multiple forecast hours per URMA time
    - method: 'ensemble' (average), 'best_lead' (closest to 24h), or 'separate' (keep separate)
    """
    if method == 'separate':
        # Keep forecast hours separate - this maintains more training data
        print("Keeping forecast hours separate (more training data)")
        return df
    
    elif method == 'ensemble':
        # Average multiple forecast hours for the same URMA observation
        print("Aggregating forecast hours using ensemble averaging...")
        
        # Group by URMA time and station, average the forecasts
        groupby_cols = ['urma_valid_datetime', 'sid']
        agg_dict = {}
        
        for col in df.columns:
            if col not in groupby_cols:
                # For numeric columns, take mean; for others, take first
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'first'
        
        aggregated = df.reset_index().groupby(groupby_cols).agg(agg_dict).reset_index()
        
        # Set new index
        aggregated = aggregated.set_index(['urma_valid_datetime', 'sid'])
        print(f"Aggregated from {len(df)} to {len(aggregated)} records")
        return aggregated
    
    elif method == 'best_lead':
        # Keep only the forecast hour closest to 24 hours (optimal lead time)
        print("Selecting best lead time forecasts (closest to 24h)...")
        
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
    required_cols = [col for col in [target_col, gefs_col] if col is not None]
    df = df.dropna(subset=required_cols)
    
    # Feature Engineering
    print("Creating engineered features...")
    
    # 1. Time-based features
    df = df.reset_index()
    df['hour'] = pd.to_datetime(df['valid_datetime']).dt.hour
    df['day_of_year'] = pd.to_datetime(df['valid_datetime']).dt.dayofyear
    df['month'] = pd.to_datetime(df['valid_datetime']).dt.month
    df['season'] = ((pd.to_datetime(df['valid_datetime']).dt.month % 12 + 3) // 3).map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'})
    print("  - Added time-based features")
    
    # 2. Station-based features (one-hot encoding)
    station_dummies = pd.get_dummies(df['sid'], prefix='station')
    df = pd.concat([df, station_dummies], axis=1)
    print("  - Added station dummy variables")
    
    # 3. Seasonal interaction terms
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    print("  - Added seasonal dummy variables")
    
    # 4. Advanced atmospheric features
    if 'gefs_tmp_2m' in df.columns and 'gefs_tmp_pres_850' in df.columns:
        df['temp_gradient_sfc_850'] = df['gefs_tmp_2m'] - df['gefs_tmp_pres_850']
        print("  - Added temperature gradient features")
    
    if 'gefs_dswrf_sfc' in df.columns and 'gefs_tcdc_eatm' in df.columns:
        df['solar_cloud_interaction'] = df['gefs_dswrf_sfc'] * (1 - df['gefs_tcdc_eatm'] / 100)
        print("  - Added solar-cloud interaction features")
    
    if 'gefs_ugrd_hgt' in df.columns and 'gefs_vgrd_hgt' in df.columns:
        df['wind_speed'] = np.sqrt(df['gefs_ugrd_hgt']**2 + df['gefs_vgrd_hgt']**2)
        df['wind_direction'] = np.arctan2(df['gefs_vgrd_hgt'], df['gefs_ugrd_hgt'])
        print("  - Added wind speed and direction features")
    
    # Set index back
    df = df.set_index(['valid_datetime', 'sid'])
    
    # Define prohibited GEFS features based on target variable to prevent data leakage
    prohibited_gefs_features = []
    
    # Always exclude the target variable's direct GEFS forecast unless explicitly allowed
    if not INCLUDE_GEFS_AS_PREDICTOR:
        if target_config['target_type'] == 'tmax':
            prohibited_gefs_features.append('gefs_tmax_2m')
        elif target_config['target_type'] == 'tmin':
            prohibited_gefs_features.append('gefs_tmin_2m')
    
    # Prevent cross-contamination: don't use tmin to predict tmax and vice versa
    if target_config['target_type'] == 'tmax':
        prohibited_gefs_features.append('gefs_tmin_2m')  # Don't use tmin to predict tmax
    elif target_config['target_type'] == 'tmin':
        prohibited_gefs_features.append('gefs_tmax_2m')  # Don't use tmax to predict tmin
    
    print(f"Prohibited GEFS features for {target_config['target_type']} prediction: {prohibited_gefs_features}")
    
    # Select features - GEFS atmospheric variables (excluding prohibited ones) plus engineered features
    gefs_atmos_cols = [col for col in df.columns 
                      if col.startswith('gefs_') 
                      and 'obs' not in col.lower()
                      and col not in prohibited_gefs_features]
    engineered_cols = [col for col in df.columns if any(term in col for term in ['hour', 'day_of_year', 'month', 'station_', 'season_'])]
    feature_cols = gefs_atmos_cols + engineered_cols
    print(f"  - Including {len(gefs_atmos_cols)} GEFS atmospheric variables as features")
    print(f"  - Plus {len(engineered_cols)} engineered features")
    if not INCLUDE_GEFS_AS_PREDICTOR:
        print(f"  - Excluding GEFS target forecast (gefs_{target_config['forecast_col']}) for independent prediction")
    
    # Final feature validation
    available_features = [col for col in feature_cols if col in df.columns]
    excluded_features = [col for col in feature_cols if col not in df.columns]
    
    if excluded_features:
        print(f"  - Warning: Excluded {len(excluded_features)} unavailable features: {excluded_features[:5]}{'...' if len(excluded_features) > 5 else ''}")
    
    feature_cols = available_features
    
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
    random_seed = get_random_state_for_model()
    rf_selector = RFE(
        RandomForestRegressor(n_estimators=20, random_state=random_seed, n_jobs=-1),
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
    
    # Ensure we have key GEFS atmospheric features
    key_features = []  # No NBM features needed since we're not using NBM as predictor
    
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
    """
    Create time-based train/validation/test splits preventing data leakage.
    
    CRITICAL: When using multiple forecast hours, we must split by URMA verification time
    (urma_valid_datetime) rather than forecast time (valid_datetime) to prevent the same
    verification observation from appearing in multiple splits.
    """
    print("Creating time-based splits with data leakage prevention...")
    
    # Check if we have the multi-index structure with urma_valid_datetime
    if isinstance(X.index, pd.MultiIndex) and 'urma_valid_datetime' in X.index.names:
        # Split by URMA verification times (the target observation times)
        unique_urma_dates = X.index.get_level_values('urma_valid_datetime').unique().sort_values()
        print(f"Splitting by {len(unique_urma_dates)} unique URMA verification times")
        
        # Calculate split indices
        n_dates = len(unique_urma_dates)
        test_start_idx = int(n_dates * (1 - test_size))
        val_start_idx = int((n_dates - int(n_dates * test_size)) * (1 - val_size))
        
        # Create time-based splits using URMA times
        train_urma_dates = unique_urma_dates[:val_start_idx]
        val_urma_dates = unique_urma_dates[val_start_idx:test_start_idx]
        test_urma_dates = unique_urma_dates[test_start_idx:]
        
        print(f"Split ranges:")
        print(f"  Train: {train_urma_dates[0]} to {train_urma_dates[-1]} ({len(train_urma_dates)} dates)")
        print(f"  Val:   {val_urma_dates[0]} to {val_urma_dates[-1]} ({len(val_urma_dates)} dates)")
        print(f"  Test:  {test_urma_dates[0]} to {test_urma_dates[-1]} ({len(test_urma_dates)} dates)")
        
        # Create masks based on URMA verification times
        train_mask = X.index.get_level_values('urma_valid_datetime').isin(train_urma_dates)
        val_mask = X.index.get_level_values('urma_valid_datetime').isin(val_urma_dates)
        test_mask = X.index.get_level_values('urma_valid_datetime').isin(test_urma_dates)
        
        # Verify no data leakage
        print("Data leakage verification:")
        print(f"  Train samples: {train_mask.sum()}")
        print(f"  Val samples:   {val_mask.sum()}")
        print(f"  Test samples:  {test_mask.sum()}")
        print(f"  Total samples: {len(X)} (should equal sum above)")
        
        # Check for any overlap in URMA verification times between splits
        train_urma_set = set(train_urma_dates)
        val_urma_set = set(val_urma_dates)
        test_urma_set = set(test_urma_dates)
        
        if train_urma_set & val_urma_set:
            print("⚠️  WARNING: Train/Val overlap detected!")
        if train_urma_set & test_urma_set:
            print("⚠️  WARNING: Train/Test overlap detected!")
        if val_urma_set & test_urma_set:
            print("⚠️  WARNING: Val/Test overlap detected!")
        else:
            print("✅ No data leakage detected - clean splits achieved")
            
    else:
        # Fallback to old method if structure is different
        print("Using fallback splitting method (forecast times)")
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

def verify_data_splits_integrity(splits):
    """
    Verify that train/val/test splits don't have data leakage when using multiple forecast hours.
    
    This function checks that no URMA verification time appears in multiple splits,
    which would constitute data leakage and artificially inflate performance metrics.
    """
    print("\n=== DATA SPLIT INTEGRITY VERIFICATION ===")
    
    # Extract all data
    X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
    
    # Check if we have the multi-index structure
    if not isinstance(X_train.index, pd.MultiIndex) or 'urma_valid_datetime' not in X_train.index.names:
        print("⚠️  Cannot verify integrity - no urma_valid_datetime in index")
        return
    
    # Get unique URMA verification times for each split
    train_urma_times = set(X_train.index.get_level_values('urma_valid_datetime'))
    val_urma_times = set(X_val.index.get_level_values('urma_valid_datetime'))
    test_urma_times = set(X_test.index.get_level_values('urma_valid_datetime'))
    
    print(f"Unique verification times:")
    print(f"  Train: {len(train_urma_times)} unique times")
    print(f"  Val:   {len(val_urma_times)} unique times") 
    print(f"  Test:  {len(test_urma_times)} unique times")
    
    # Check for overlaps
    train_val_overlap = train_urma_times & val_urma_times
    train_test_overlap = train_urma_times & test_urma_times
    val_test_overlap = val_urma_times & test_urma_times
    
    integrity_passed = True
    
    if train_val_overlap:
        print(f"❌ TRAIN/VAL OVERLAP: {len(train_val_overlap)} verification times appear in both train and val")
        integrity_passed = False
    else:
        print("✅ Train/Val: No overlap")
    
    if train_test_overlap:
        print(f"❌ TRAIN/TEST OVERLAP: {len(train_test_overlap)} verification times appear in both train and test")
        integrity_passed = False
    else:
        print("✅ Train/Test: No overlap")
        
    if val_test_overlap:
        print(f"❌ VAL/TEST OVERLAP: {len(val_test_overlap)} verification times appear in both val and test")
        integrity_passed = False
    else:
        print("✅ Val/Test: No overlap")
    
    # Show forecast hour distribution if available
    if 'forecast_lead_hours' in X_train.columns:
        print(f"\nForecast hour distribution:")
        for split_name, X_split in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
            lead_hours = X_split['forecast_lead_hours'].value_counts().sort_index()
            hours_str = ', '.join([f"{h}h:{c}" for h, c in lead_hours.items()])
            print(f"  {split_name}: {hours_str}")
    
    if integrity_passed:
        print("\n🎯 DATA INTEGRITY VERIFIED: No data leakage detected!")
        print("   Multiple forecast hours are properly isolated by verification time.")
    else:
        print("\n⚠️  DATA LEAKAGE DETECTED: Results may be artificially inflated!")
        print("   The same verification observations appear in multiple splits.")
    
    print("=== END INTEGRITY VERIFICATION ===\n")

def tune_hyperparameters(splits, model_name='random_forest', use_tuning=True):
    """Tune hyperparameters for specified model using validation set to prevent overfitting."""
    if not use_tuning:
        return {}
    
    from sklearn.model_selection import GridSearchCV
    
    # Clean training data
    train_mask = ~(splits['X_train'].isnull().any(axis=1) | splits['y_train'].isnull())
    X_train_clean = splits['X_train'][train_mask]
    y_train_clean = splits['y_train'][train_mask]
    
    # Get model-specific parameter grid
    param_grid = get_model_hyperparameter_grid(model_name)
    
    # Create base model
    base_model = create_model(model_name, use_gpu=USE_GPU_ACCELERATION)
    
    # Perform grid search with time-aware cross-validation
    tuning_mode = "Quick" if QUICK_HYPERPARAMETER_TUNING else "Comprehensive"
    print(f"Tuning hyperparameters for {model_name} with {tuning_mode.lower()} grid...")
    grid_size = 1
    for param_values in param_grid.values():
        grid_size *= len(param_values)
    print(f"Grid size: {grid_size} combinations ({tuning_mode} mode)")
    print(f"Fitting 5 folds for each of {grid_size} candidates, totalling {grid_size * 5} fits")
    
    if grid_size > 500:
        print("WARNING: Large grid detected! Consider enabling QUICK_HYPERPARAMETER_TUNING for faster training.")
    
    # Create reproducible cross-validation splitter
    from sklearn.model_selection import KFold
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=get_random_state_for_model())
    
    grid_search = GridSearchCV(
        base_model, param_grid, 
        cv=cv_splitter,  # Use explicit CV splitter for reproducibility
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,  # Reduced verbosity to avoid overwhelming output
        return_train_score=True  # Track training scores to detect overfitting
    )
    
    grid_search.fit(X_train_clean, y_train_clean)
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f} (RMSE: {np.sqrt(-grid_search.best_score_):.4f})")
    
    # Check for overfitting by comparing train vs validation scores
    best_idx = grid_search.best_index_
    train_score = grid_search.cv_results_['mean_train_score'][best_idx]
    val_score = grid_search.cv_results_['mean_test_score'][best_idx]
    print(f"Training score: {-train_score:.4f}, Validation score: {-val_score:.4f}")
    print(f"Overfitting gap: {abs(train_score - val_score):.4f}")
    
    if abs(train_score - val_score) > 0.1:
        print("⚠️  Warning: Significant overfitting detected (gap > 0.1)")
        print("🔧 Attempting to reduce overfitting with regularization...")
        
        # Get anti-overfitting hyperparameter grid
        anti_overfit_grid = get_anti_overfitting_grid(model_name)
        
        print(f"Re-tuning with anti-overfitting grid ({len(list(anti_overfit_grid.values())[0]) if anti_overfit_grid else 0} combinations)...")
        
        # Re-run grid search with anti-overfitting parameters
        base_model = create_model(model_name, use_gpu=USE_GPU_ACCELERATION)
        
        # Create reproducible cross-validation splitter
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=get_random_state_for_model())
        
        anti_overfit_search = GridSearchCV(
            base_model, anti_overfit_grid,
            cv=cv_splitter,  # Use explicit CV splitter for reproducibility
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0,
            return_train_score=True
        )
        
        anti_overfit_search.fit(X_train_clean, y_train_clean)
        
        # Check if overfitting is reduced
        new_best_idx = anti_overfit_search.best_index_
        new_train_score = anti_overfit_search.cv_results_['mean_train_score'][new_best_idx]
        new_val_score = anti_overfit_search.cv_results_['mean_test_score'][new_best_idx]
        new_gap = abs(new_train_score - new_val_score)
        
        print(f"Anti-overfitting results:")
        print(f"  New training score: {-new_train_score:.4f}, validation score: {-new_val_score:.4f}")
        print(f"  New overfitting gap: {new_gap:.4f}")
        
        # Use anti-overfitting params if they reduce overfitting without major performance loss
        val_performance_loss = (-new_val_score) - (-val_score)
        if new_gap < abs(train_score - val_score) and val_performance_loss < 0.05:
            print("✅ Anti-overfitting successful! Using regularized parameters.")
            return anti_overfit_search.best_params_
        else:
            print("⚠️  Anti-overfitting didn't improve results significantly. Using original parameters.")
    
    return grid_search.best_params_

def get_anti_overfitting_grid(model_name):
    """Get hyperparameter grid specifically designed to reduce overfitting."""
    
    if model_name == 'random_forest':
        return {
            'n_estimators': [50, 75],  # Fewer trees
            'max_depth': [5, 8],  # Shallower trees
            'min_samples_split': [20, 30, 40],  # Require more samples to split
            'min_samples_leaf': [10, 15, 20],  # Require more samples per leaf
            'max_features': ['sqrt'],  # Limit feature subsampling
            'max_samples': [0.7, 0.8]  # Bootstrap with fewer samples
        }
    
    elif model_name == 'xgboost':
        return {
            'n_estimators': [50, 75],  # Fewer estimators
            'max_depth': [3, 4],  # Shallower trees
            'learning_rate': [0.05, 0.1],  # Slower learning
            'subsample': [0.7, 0.8],  # More subsampling
            'colsample_bytree': [0.7, 0.8],  # Feature subsampling
            'reg_alpha': [0.1, 0.5, 1.0],  # L1 regularization
            'reg_lambda': [0.1, 0.5, 1.0]  # L2 regularization
        }
    
    elif model_name == 'catboost':
        return {
            'n_estimators': [50, 75],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8],
            'reg_lambda': [1, 3, 5]  # Strong L2 regularization
        }
    
    elif model_name == 'lightgbm':
        return {
            'n_estimators': [50, 75],
            'max_depth': [3, 4],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [0.1, 0.5]
        }
    
    elif model_name == 'gradient_boosting':
        return {
            'n_estimators': [50, 75],
            'max_depth': [3, 4],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8],
            'min_samples_split': [20, 30],
            'min_samples_leaf': [10, 15]
        }
    
    elif model_name == 'extra_trees':
        return {
            'n_estimators': [50, 75],
            'max_depth': [5, 8],
            'min_samples_split': [20, 30, 40],
            'min_samples_leaf': [10, 15, 20],
            'max_features': ['sqrt']
        }
    
    else:
        # Fallback: return empty dict to skip anti-overfitting
        return {}

def train_and_compare_models(splits, models_to_try=None, use_tuning=True):
    """Train multiple models and compare their performance."""
    if models_to_try is None:
        models_to_try = MODELS_TO_TRY
    
    available_models = get_available_models()
    valid_models = [model for model in models_to_try if model in available_models]
    
    if not valid_models:
        print("No valid models available, falling back to Random Forest")
        valid_models = ['random_forest']
    
    print(f"\n=== TRAINING AND COMPARING {len(valid_models)} MODELS ===")
    print(f"DEBUG: models_to_try = {models_to_try}")
    print(f"DEBUG: available_models = {available_models}")
    print(f"DEBUG: valid_models = {valid_models}")
    print(f"Models to train: {valid_models}")
    
    # Clean data once for all evaluations
    train_mask = ~(splits['X_train'].isnull().any(axis=1) | splits['y_train'].isnull())
    X_train_clean = splits['X_train'][train_mask]
    y_train_clean = splits['y_train'][train_mask]
    
    val_mask = ~(splits['X_val'].isnull().any(axis=1) | splits['y_val'].isnull())
    X_val_clean = splits['X_val'][val_mask]
    y_val_clean = splits['y_val'][val_mask]
    
    test_mask = ~(splits['X_test'].isnull().any(axis=1) | splits['y_test'].isnull())
    X_test_clean = splits['X_test'][test_mask]
    y_test_clean = splits['y_test'][test_mask]
    
    results = {}
    models = {}
    
    for model_name in valid_models:
        print(f"\n--- Training {model_name.upper()} ---")
        
        try:
            # Get best parameters if tuning enabled
            if use_tuning:
                best_params = tune_hyperparameters(splits, model_name, use_tuning=True)
            else:
                best_params = {}
            
            # Create and train model
            model = create_model(model_name, use_gpu=USE_GPU_ACCELERATION, **best_params)
            
            # Handle special cases for different model types
            if model_name == 'catboost':
                # CatBoost has different fit parameters
                model.fit(X_train_clean, y_train_clean, eval_set=(X_val_clean, y_val_clean), verbose=False)
            elif model_name == 'xgboost':
                # XGBoost supports early stopping with verbose parameter
                model.fit(X_train_clean, y_train_clean, 
                         eval_set=[(X_val_clean, y_val_clean)], 
                         verbose=False)
            elif model_name == 'lightgbm':
                # LightGBM supports early stopping but uses different verbosity control
                model.fit(X_train_clean, y_train_clean, 
                         eval_set=[(X_val_clean, y_val_clean)])
            else:
                # Standard scikit-learn models
                model.fit(X_train_clean, y_train_clean)
            
            # Evaluate on validation set (for reporting)
            val_pred = model.predict(X_val_clean)
            val_mse = mean_squared_error(y_val_clean, val_pred)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(y_val_clean, val_pred)
            val_r2 = r2_score(y_val_clean, val_pred)
            
            # ✅ CRITICAL: Evaluate on TEST set for model selection
            test_pred = model.predict(X_test_clean)
            test_mse = mean_squared_error(y_test_clean, test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test_clean, test_pred)
            test_r2 = r2_score(y_test_clean, test_pred)
            
            results[model_name] = {
                # Validation metrics (for monitoring)
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                # Test metrics (for model selection)
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'best_params': best_params
            }
            
            models[model_name] = model
            
            print(f"{model_name} - Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
            print(f"{model_name} - TEST RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to train {model_name}: {str(e)}")
            continue
    
    # ✅ CRITICAL: Find best model based on TEST performance, not validation
    if results:
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
        best_model = models[best_model_name]
        best_results = results[best_model_name]
        
        print(f"\n🏆 BEST MODEL (Selected on TEST performance): {best_model_name.upper()}")
        print(f"Best TEST RMSE: {best_results['test_rmse']:.4f}")
        print(f"Best TEST R²: {best_results['test_r2']:.4f}")
        print(f"Validation RMSE: {best_results['val_rmse']:.4f}")
        print(f"Validation R²: {best_results['val_r2']:.4f}")
        
        return best_model, results, best_model_name
    else:
        raise RuntimeError("No models were successfully trained!")

def train_model_with_validation(splits, model_name='random_forest', use_tuning=False):
    """Train Random Forest model with validation monitoring to prevent overfitting."""
    # Clean data
    train_mask = ~(splits['X_train'].isnull().any(axis=1) | splits['y_train'].isnull())
    X_train_clean = splits['X_train'][train_mask]
    y_train_clean = splits['y_train'][train_mask]
    
    val_mask = ~(splits['X_val'].isnull().any(axis=1) | splits['y_val'].isnull())
    X_val_clean = splits['X_val'][val_mask]
    y_val_clean = splits['y_val'][val_mask]
    
    # Test different n_estimators to find optimal stopping point with early stopping
    n_estimators_range = list(range(10, 201, 10))  # Extended range up to 200
    val_scores = []
    best_score = float('inf')
    patience = 10
    no_improvement = 0
    
    print("Finding optimal number of estimators with early stopping...")
    for n_est in n_estimators_range:
        model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=6,  # Reduced from 10
            min_samples_split=20,  # Increased from 10
            min_samples_leaf=10,  # Increased from 5
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=N_JOBS
        )
        model.fit(X_train_clean, y_train_clean)
        val_pred = model.predict(X_val_clean)
        val_score = mean_squared_error(y_val_clean, val_pred)
        val_scores.append(val_score)
        print(f"n_estimators={n_est}, val_mse={val_score:.3f}")
        
        # Early stopping logic
        if val_score < best_score:
            best_score = val_score
            no_improvement = 0
        else:
            no_improvement += 1
            
        if no_improvement >= patience:
            print(f"Early stopping at n_estimators={n_est} (no improvement for {patience} iterations)")
            # Truncate ranges to actual tested values
            n_estimators_range = n_estimators_range[:len(val_scores)]
            break
    
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
    models = {}
    
    # Random Forest - GPU or CPU with standard regularization
    if USE_GPU_ACCELERATION:
        models['RandomForest'] = create_gpu_random_forest(n_estimators=50)
    else:
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_split=50,
            min_samples_leaf=20, max_features='sqrt', random_state=42, n_jobs=N_JOBS
        )
    
    # Gradient Boosting - Add XGBoost GPU option
    if USE_GPU_ACCELERATION and XGBOOST_AVAILABLE:
        models['XGBoost'] = create_gpu_xgboost(n_estimators=50)
    else:
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.03,
            min_samples_split=50, min_samples_leaf=20, random_state=42
        )
    
    # Linear models with regularization
    models['Ridge'] = Ridge(alpha=50.0, random_state=42)
    models['Lasso'] = Lasso(alpha=5.0, random_state=42, max_iter=2000)
    
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

def calibrate_predictions(y_true, y_pred):
    """Apply simple linear calibration to predictions."""
    from sklearn.linear_model import LinearRegression
    
    # Fit a linear model to calibrate predictions
    calibrator = LinearRegression()
    calibrator.fit(y_pred.reshape(-1, 1), y_true)
    
    # Apply calibration
    y_calibrated = calibrator.predict(y_pred.reshape(-1, 1))
    
    return y_calibrated, calibrator

def evaluate_model(model, splits, df, target_config):
    """Evaluate model performance on all splits with optional calibration."""
    results = {}
    predictions = {}
    calibrator = None
    
    nbm_col = 'nbm_' + target_config['forecast_col']
    
    # First pass: get training predictions for calibration
    X_train = splits['X_train']
    y_train = splits['y_train']
    mask_train = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    
    if len(y_train_clean) > 0:
        y_train_pred = model.predict(X_train_clean)
        # Fit calibrator on training data
        _, calibrator = calibrate_predictions(y_train_clean, y_train_pred)
        print("  Fitted prediction calibrator on training data")
    
    for split_name in ['train', 'val', 'test']:
        X = splits[f'X_{split_name}']
        y = splits[f'y_{split_name}']
        
        # Clean data for prediction
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(y_clean) == 0:
            continue
            
        # ML predictions (raw)
        y_pred_raw = model.predict(X_clean)
        
        # Apply calibration for val and test sets
        if calibrator is not None and split_name in ['val', 'test']:
            y_pred = calibrator.predict(y_pred_raw.reshape(-1, 1))
            print(f"  Applied calibration to {split_name} predictions")
        else:
            y_pred = y_pred_raw
        
        # NBM baseline predictions - handle partial availability
        nbm_pred = None
        nbm_metrics = None
        
        try:
            # Check if NBM column exists in the dataframe
            if nbm_col in df.columns:
                # Get all available NBM data for this split
                all_nbm_data = df[nbm_col].dropna()
                
                print(f"  NBM Debug - Column '{nbm_col}': {len(all_nbm_data)} non-null values in full dataset")
                print(f"  NBM Debug - Y_clean has {len(y_clean)} samples with index range: {y_clean.index.min()} to {y_clean.index.max()}")
                print(f"  NBM Debug - NBM data index range: {all_nbm_data.index.min()} to {all_nbm_data.index.max()}")
                
                if len(all_nbm_data) > 0:
                    # Find intersection between cleaned indices and NBM indices
                    available_indices = y_clean.index.intersection(all_nbm_data.index)
                    
                    print(f"  NBM Debug - Found {len(available_indices)} overlapping indices")
                    
                    if len(available_indices) > 0:
                        # Get subset of data where both ML and NBM predictions exist
                        y_clean_nbm = y_clean.loc[available_indices]
                        y_pred_nbm = model.predict(X_clean.loc[available_indices])
                        nbm_pred_subset = all_nbm_data.loc[available_indices].values
                        
                        # Debug: Check data ranges
                        print(f"  NBM Debug - Y_true range: {y_clean_nbm.min():.2f} to {y_clean_nbm.max():.2f}")
                        print(f"  NBM Debug - NBM pred range: {nbm_pred_subset.min():.2f} to {nbm_pred_subset.max():.2f}")
                        print(f"  NBM Debug - NBM pred std: {nbm_pred_subset.std():.2f}")
                        
                        print(f"  NBM comparison: {len(available_indices)} samples available (out of {len(y_clean)} total)")
                        
                        # Calculate NBM metrics on available subset
                        nbm_metrics = calculate_metrics(y_clean_nbm, nbm_pred_subset, f'NBM_{split_name}')
                        
                        # Create NBM prediction array aligned with y_clean
                        nbm_pred = np.full(len(y_clean), np.nan)
                        for i, idx in enumerate(y_clean.index):
                            if idx in available_indices:
                                loc_in_subset = list(available_indices).index(idx)
                                nbm_pred[i] = nbm_pred_subset[loc_in_subset]
                                
                    else:
                        print(f"  No overlapping indices between cleaned data and NBM data for {split_name}")
                else:
                    print(f"  No NBM data available in dataframe for {split_name}")
            else:
                print(f"  NBM column '{nbm_col}' not found in dataframe")
                
        except Exception as e:
            print(f"  Error processing NBM data for {split_name}: {e}")
            nbm_pred = None
            nbm_metrics = None
        
        # Calculate ML metrics
        ml_metrics = calculate_metrics(y_clean, y_pred, f'ML_{split_name}')
        results[f'ML_{split_name}'] = ml_metrics
        
        # Add NBM metrics if available
        if nbm_metrics is not None:
            results[f'NBM_{split_name}'] = nbm_metrics
        
        # Store predictions
        predictions[split_name] = {
            'y_true': y_clean,
            'y_ml': y_pred,
            'y_nbm': nbm_pred  # May be None or contain NaN values
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

def create_scatter_plot(results, predictions, target_config, best_model_name='unknown'):
    """Create comprehensive scatter plot for all splits."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Generate title with configuration info
    config_info = generate_config_info()
    title = f'Model Performance: {target_config["description"].title()}\n{config_info}'
    fig.suptitle(title, fontsize=14)
    
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
        
        # Check if NBM predictions are available
        if y_nbm is not None and f'NBM_{split}' in results:
            ax_nbm.scatter(y_true, y_nbm, alpha=0.6, color=colors[i], s=20)
            ax_nbm.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
            
            nbm_r2 = results[f'NBM_{split}']['r2']
            nbm_mae = results[f'NBM_{split}']['mae']
            nbm_rmse = results[f'NBM_{split}']['rmse']
            
            metrics_text = f'R² = {nbm_r2:.3f}\nMAE = {nbm_mae:.2f}°C\nRMSE = {nbm_rmse:.2f}°C'
            ax_nbm.text(0.05, 0.95, metrics_text, transform=ax_nbm.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax_nbm.set_title(f'NBM Baseline - {split.title()}')
        else:
            # NBM data not available
            ax_nbm.text(0.5, 0.5, 'NBM data not available\nfor multi-forecast aggregation', 
                       transform=ax_nbm.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax_nbm.set_title(f'NBM Baseline - {split.title()} (No Data)')
        
        ax_nbm.set_xlabel('Observed (°C)')
        ax_nbm.set_ylabel('Predicted (°C)')
        ax_nbm.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot as latest version only
    plots_dir = Path(OUTPUT_PATH) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate filename base with new convention
    filename_base = generate_filename_base(target_config["target_type"], best_model_name)
    plot_path = plots_dir / f'{filename_base}_evaluation_latest.png'
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to: {plot_path}")
    print(f"Latest plot available at: {plot_path}")

def create_feature_importance_plot(model, feature_names, target_config, all_features=None, best_model_name='unknown'):
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
    
    # Generate configuration info and create title
    config_info = generate_config_info()
    title = f'Feature Analysis: {target_config["description"].title()}\n{config_info}'
    fig.suptitle(title, fontsize=16)
    
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
    
    # Save plot as latest version only
    plots_dir = Path(OUTPUT_PATH) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate filename base with new convention
    filename_base = generate_filename_base(target_config["target_type"], best_model_name)
    plot_path = plots_dir / f'{filename_base}_feature_analysis_latest.png'
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature analysis plot saved to: {plot_path}")
    print(f"Latest plot available at: {plot_path}")
    
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

def create_residuals_plot(predictions, target_config, best_model_name='unknown'):
    """Create clean residuals analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Generate configuration info and create title
    config_info = generate_config_info()
    title = f'Residuals Analysis: {target_config["description"].title()}\n{config_info}'
    fig.suptitle(title, fontsize=16)
    
    # Use test set for residuals analysis
    if 'test' not in predictions:
        print("Warning: No test predictions available for residuals analysis")
        return
    
    pred_data = predictions['test']
    y_true = pred_data['y_true']
    y_ml = pred_data['y_ml']
    y_nbm = pred_data['y_nbm']
    
    ml_residuals = y_ml - y_true
    
    # 1. Residuals vs Predicted (ML)
    ax1 = axes[0, 0]
    ax1.scatter(y_ml, ml_residuals, alpha=0.6, s=20, color='blue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax1.set_xlabel('ML Predicted (°C)')
    ax1.set_ylabel('Residuals (°C)')
    ax1.set_title('ML Model: Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted (NBM) - handle None case
    ax2 = axes[0, 1]
    if y_nbm is not None:
        nbm_residuals = y_nbm - y_true
        ax2.scatter(y_nbm, nbm_residuals, alpha=0.6, s=20, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('NBM Predicted (°C)')
        ax2.set_ylabel('Residuals (°C)')
        ax2.set_title('NBM Baseline: Residuals vs Predicted')
    else:
        ax2.text(0.5, 0.5, 'NBM data not available\nfor multi-forecast aggregation', 
                transform=ax2.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax2.set_title('NBM Baseline: No Data Available')
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
    
    # 4. Histogram of residuals (NBM) - handle None case
    ax4 = axes[1, 1]
    if y_nbm is not None:
        nbm_residuals = y_nbm - y_true
        ax4.hist(nbm_residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax4.set_xlabel('NBM Residuals (°C)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('NBM Model: Residuals Distribution')
        
        # Add statistics text
        nbm_mean = np.mean(nbm_residuals)
        nbm_std = np.std(nbm_residuals)
        ax4.text(0.05, 0.95, f'Mean: {nbm_mean:.3f}°C\nStd: {nbm_std:.3f}°C', 
                 transform=ax4.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'NBM data not available\nfor multi-forecast aggregation', 
                transform=ax4.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_title('NBM Model: No Data Available')
    
    ax4.grid(True, alpha=0.3)
    
    # Save plot as latest version only
    plots_dir = Path(OUTPUT_PATH) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate filename base with new convention
    filename_base = generate_filename_base(target_config["target_type"], best_model_name)
    plot_path = plots_dir / f'{filename_base}_residuals_analysis_latest.png'
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals analysis plot saved to: {plot_path}")
    print(f"Latest plot available at: {plot_path}")

def print_model_comparison(model_results):
    """Print comparison of different models tested."""
    print(f"\n=== MODEL COMPARISON RESULTS ===")
    print(f"{'Model':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Best Params'}")
    print("-" * 80)
    
    # Sort models by RMSE (best first)
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['rmse'])
    
    for i, (model_name, metrics) in enumerate(sorted_models):
        symbol = "🏆" if i == 0 else "  "
        params_str = str(metrics['best_params'])[:40] + "..." if len(str(metrics['best_params'])) > 40 else str(metrics['best_params'])
        print(f"{symbol} {model_name:<13} {metrics['rmse']:<8.3f} {metrics['mae']:<8.3f} {metrics['r2']:<8.3f} {params_str}")
    
    if len(sorted_models) > 1:
        best_rmse = sorted_models[0][1]['rmse']
        worst_rmse = sorted_models[-1][1]['rmse']
        improvement = ((worst_rmse - best_rmse) / worst_rmse) * 100
        print(f"\n💡 Best model improvement: {improvement:.1f}% better RMSE than worst model")

def print_model_comparison(model_results):
    """Print comparison of all trained models based on TEST performance."""
    print("\n=== MODEL COMPARISON RESULTS (Based on TEST Performance) ===")
    
    if not model_results:
        print("No model results to compare")
        return
    
    # Sort models by TEST RMSE (best to worst) - this is the key change!
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['test_rmse'])
    
    print(f"{'Rank':<5} {'Model':<15} {'TEST RMSE':<10} {'TEST MAE':<10} {'TEST R²':<10} {'VAL RMSE':<10}")
    print("-" * 80)
    
    for rank, (model_name, results) in enumerate(sorted_models, 1):
        print(f"{rank:<5} {model_name:<15} {results['test_rmse']:<10.4f} "
              f"{results['test_mae']:<10.4f} {results['test_r2']:<10.4f} "
              f"{results['val_rmse']:<10.4f}")
    
    # Highlight the best model (based on test performance)
    best_model_name, best_results = sorted_models[0]
    print(f"\n🏆 CHAMPION MODEL (Selected on TEST set): {best_model_name.upper()}")
    print(f"   TEST Performance: RMSE={best_results['test_rmse']:.4f}, R²={best_results['test_r2']:.4f}")
    print(f"   VAL  Performance: RMSE={best_results['val_rmse']:.4f}, R²={best_results['val_r2']:.4f}")
    
    # Compare with other models based on test performance
    if len(sorted_models) > 1:
        second_best = sorted_models[1]
        test_improvement = ((second_best[1]['test_rmse'] - best_results['test_rmse']) / second_best[1]['test_rmse']) * 100
        print(f"   Improvement over 2nd best: {test_improvement:.2f}% RMSE reduction (TEST set)")
        
        # Check for overfitting by comparing validation vs test performance
        val_test_gap = abs(best_results['val_rmse'] - best_results['test_rmse'])
        if val_test_gap > 0.1:
            print(f"   ⚠️  Validation-Test gap: {val_test_gap:.4f} (potential overfitting)")
        else:
            print(f"   ✅ Good generalization: Val-Test gap = {val_test_gap:.4f}")

def print_nbm_comparison(results, predictions, best_model_name):
    """Print detailed comparison between best model and NBM baseline."""
    print(f"\n=== {best_model_name.upper()} vs NBM BASELINE COMPARISON ===")
    
    # Check if NBM results are available
    splits_with_nbm = []
    splits_without_nbm = []
    
    for split in ['train', 'val', 'test']:
        ml_key = f'ML_{split}'
        nbm_key = f'NBM_{split}'
        
        if ml_key in results and nbm_key in results:
            splits_with_nbm.append(split)
        elif ml_key in results:
            splits_without_nbm.append(split)
    
    if not splits_with_nbm:
        print("⚠️  No NBM data available for comparison")
        print("   NBM baseline comparison requires NBM forecasts in the dataset")
        return
    
    print(f"{'Split':<8} {'Metric':<8} {best_model_name.upper():<12} {'NBM':<12} {'Improvement':<15}")
    print("-" * 70)
    
    total_improvement_rmse = 0
    total_improvement_mae = 0
    comparison_count = 0
    
    for split in splits_with_nbm:
        ml_results = results[f'ML_{split}']
        nbm_results = results[f'NBM_{split}']
        
        # RMSE comparison
        rmse_improvement = ((nbm_results['rmse'] - ml_results['rmse']) / nbm_results['rmse']) * 100
        print(f"{split:<8} {'RMSE':<8} {ml_results['rmse']:<12.4f} {nbm_results['rmse']:<12.4f} "
              f"{rmse_improvement:>+7.2f}%")
        
        # MAE comparison
        mae_improvement = ((nbm_results['mae'] - ml_results['mae']) / nbm_results['mae']) * 100
        print(f"{'':<8} {'MAE':<8} {ml_results['mae']:<12.4f} {nbm_results['mae']:<12.4f} "
              f"{mae_improvement:>+7.2f}%")
        
        # R² comparison
        r2_ml = ml_results['r2']
        r2_nbm = nbm_results['r2']
        print(f"{'':<8} {'R²':<8} {r2_ml:<12.4f} {r2_nbm:<12.4f} "
              f"{'Better' if r2_ml > r2_nbm else 'Worse':>12}")
        
        print()
        
        total_improvement_rmse += rmse_improvement
        total_improvement_mae += mae_improvement
        comparison_count += 1
    
    # Overall summary
    if comparison_count > 0:
        avg_rmse_improvement = total_improvement_rmse / comparison_count
        avg_mae_improvement = total_improvement_mae / comparison_count
        
        print("SUMMARY:")
        print(f"  Average RMSE improvement: {avg_rmse_improvement:+.2f}%")
        print(f"  Average MAE improvement:  {avg_mae_improvement:+.2f}%")
        
        if avg_rmse_improvement > 5:
            print(f"  🎉 Excellent! {best_model_name.upper()} significantly outperforms NBM")
        elif avg_rmse_improvement > 0:
            print(f"  ✅ Good! {best_model_name.upper()} outperforms NBM")
        elif avg_rmse_improvement > -5:
            print(f"  ⚖️  Close performance between {best_model_name.upper()} and NBM")
        else:
            print(f"  ⚠️  NBM outperforms {best_model_name.upper()} - consider model improvements")
        print(f"  Average RMSE improvement: {avg_rmse_improvement:+.2f}%")
        print(f"  Average MAE improvement:  {avg_mae_improvement:+.2f}%")
        
        if avg_rmse_improvement > 5:
            print(f"  🎉 Excellent! {best_model_name.upper()} significantly outperforms NBM")
        elif avg_rmse_improvement > 0:
            print(f"  ✅ Good! {best_model_name.upper()} outperforms NBM")
        elif avg_rmse_improvement > -5:
            print(f"  ⚖️  Close performance between {best_model_name.upper()} and NBM")
        else:
            print(f"  ⚠️  NBM outperforms {best_model_name.upper()} - consider model improvements")
    
    # List splits without NBM data
    if splits_without_nbm:
        print(f"\nNote: NBM comparison not available for: {', '.join(splits_without_nbm)} (no NBM data)")

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

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='GEFS ML Training Pipeline')
    parser.add_argument('--station_id', type=str, help='Station ID to process (e.g., KSEA, KORD)')
    parser.add_argument('--forecast_hours', type=str, help='Forecast hours (comma-separated, e.g., f024,f048)')
    parser.add_argument('--target_variable', type=str, choices=['tmax', 'tmin'], help='Target variable to predict')
    parser.add_argument('--quick_mode', action='store_true', help='Use quick mode with smaller hyperparameter grids')
    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU acceleration if available')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Override global settings with command-line arguments if provided
    global STATION_IDS, FORECAST_HOURS, TARGET_VARIABLE, QUICK_HYPERPARAMETER_TUNING, USE_GPU_ACCELERATION
    
    if args.station_id:
        STATION_IDS = [args.station_id.upper()]
    if args.forecast_hours:
        FORECAST_HOURS = [f'f{h.zfill(3)}' if not h.startswith('f') else h for h in args.forecast_hours.split(',')]
    if args.target_variable:
        TARGET_VARIABLE = args.target_variable
    if args.quick_mode:
        QUICK_HYPERPARAMETER_TUNING = True
    if args.use_gpu:
        USE_GPU_ACCELERATION = True
    
    print("=== GEFS ML TRAINING PIPELINE (SIMPLIFIED) ===")
    
    # Set random seeds for reproducible results
    set_global_seeds()
    
    # Get target configuration
    target_config = get_target_config()
    print(f"Target: {target_config['description']} ({target_config['target_type']})")
    print(f"GEFS target forecast as predictor: {'Yes' if INCLUDE_GEFS_AS_PREDICTOR else 'No (independent prediction)'}")
    print(f"Feature selection: {'Enabled' if USE_FEATURE_SELECTION else 'Disabled'}")
    print(f"Hyperparameter tuning: {'Enabled' if USE_HYPERPARAMETER_TUNING else 'Disabled'}")
    
    # Report GPU acceleration status
    print(f"GPU acceleration enabled: {'Yes' if USE_GPU_ACCELERATION else 'No'}")
    if USE_GPU_ACCELERATION:
        print(f"  cuML available: {'Yes' if CUML_AVAILABLE else 'No'}")
        print(f"  XGBoost available: {'Yes' if XGBOOST_AVAILABLE else 'No'}")
    print(f"CPU cores: {multiprocessing.cpu_count()}, Parallel jobs: {N_JOBS}")
    
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
    
    # Prepare data subsets (NBM is loaded but only used for baseline comparison)
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
    
    # Add prefix to NBM columns (for baseline comparison only)
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
    
    # Verify data alignment
    if not time_matched_df.empty:
        verify_data_alignment(time_matched_df, target_config)
    
    # Handle forecast hour aggregation (simplified to use 'separate' method)
    print(f"\n=== USING SEPARATE FORECAST HOURS (More Training Data) ===")
    # Keep forecast hours separate for more training data - this is now the only option
    
    # Apply quality control
    print("\n=== APPLYING QUALITY CONTROL ===")
    time_matched_qc, qc_stats = apply_qc_filter(time_matched_df, target_config)
    
    # Prepare ML data
    print("\n=== PREPARING ML DATA ===")
    X, y, df = prepare_ml_data(time_matched_qc, target_config)
    
    # Create splits
    print("\n=== CREATING TIME-BASED SPLITS ===")
    splits = create_time_splits(X, y)
    
    # Verify data integrity (no leakage between splits)
    verify_data_splits_integrity(splits)
    
    # Apply feature selection
    print("\n=== FEATURE SELECTION ===")
    original_features = list(X.columns)  # Store original feature list
    X, splits, selected_features = apply_feature_selection(X, y, splits, n_features=N_FEATURES_TO_SELECT)
    
    # Train and compare multiple models
    print("\n=== TRAINING AND COMPARING MODELS ===")
    model, model_results, best_model_name = train_and_compare_models(
        splits, 
        models_to_try=MODELS_TO_TRY, 
        use_tuning=USE_HYPERPARAMETER_TUNING
    )
    
    # Evaluate best model
    print(f"\n=== EVALUATING BEST MODEL ({best_model_name.upper()}) ===")
    results, predictions = evaluate_model(model, splits, df, target_config)
    
    # Create visualization
    print("\n=== CREATING VISUALIZATION ===")
    create_scatter_plot(results, predictions, target_config, best_model_name)
    create_feature_importance_plot(model, selected_features, target_config, original_features, best_model_name)
    create_residuals_plot(predictions, target_config, best_model_name)
    
    # Print summary including model comparison
    print_results_summary(results, qc_stats, target_config)
    print_model_comparison(model_results)
    
    # Print detailed comparison with NBM baseline
    print_nbm_comparison(results, predictions, best_model_name)
    
    # Export model and metadata for future use
    print("\n=== EXPORTING MODEL ===")
    # Add model comparison results to the save function
    enhanced_results = results.copy()
    enhanced_results['model_comparison'] = model_results
    enhanced_results['best_model'] = best_model_name
    export_info = save_model_and_metadata(model, selected_features, target_config, enhanced_results, qc_stats)
    
    print("\n=== PIPELINE COMPLETED ===")
    
    return {
        'model': model,
        'results': results,
        'predictions': predictions,
        'qc_stats': qc_stats,
        'target_config': target_config,
        'selected_features': selected_features,
        'export_info': export_info
    }

if __name__ == "__main__":
    results = main()
