#!/home/chad.kahler/anaconda3/envs/dev/bin/python
"""
Weather forecasting script with quality control for raw observations, using GEFS data for training
and NBM data for scatter plot comparison, for a single dynamically specified station.

This script processes weather station data for a single station (specified by process_station) from
GEFS and NBM, applies quality control using IQR outlier detection and temporal consistency checks,
performs PCA, and trains an XGBoost model to predict maximum temperature (tmax_obs). It generates a
single 4-panel scatter plot comparing raw GEFS and NBM tmax_2m vs tmax_obs, and training/testing
predictions, with metrics (R², MAE, RMSE) displayed on each panel. All labels and file names
dynamically adjust based on the process_station variable.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import joblib
from datetime import datetime, timedelta
from xgboost import XGBRegressor
import optuna
from optuna.pruners import MedianPruner

# Configuration constants
STATIONS_FILE = '/home/chad.kahler/gefs/wr_obs_complete_index.csv'
GEFS_START_DATE = "2000-01-01"  # Start date for GEFS analysis period
GEFS_END_DATE = "2009-12-31"    # End date for GEFS analysis period
TARGET_VARIABLE = "tmax_obs"    # Target variable for prediction
FORECAST_HOURS = [24]           # Forecast hours to process
GEFS_DATA_DIR = "/nas/stid/data/gefs_reforecast/station_data/combined"  # GEFS input station data directory
NBM_DATA_DIR = "/nas/stid/data/nbm/station_data/combined"    # NBM input station data directory
PROCESS_STATION = "KSLC"        # Modified: Single station ID, change this to any station (e.g., "KSEA")
BASE_OUTPUT_DIR = f"./results_{PROCESS_STATION}_single_fcst_hour"  # Modified: Dynamic output directory based on station
TARGET_CORR_THRESHOLD = 1.1     # Threshold for excluding highly correlated variables
LOADING_THRESHOLD = 0.1         # Threshold for PCA loading plots
VARIANCE_THRESHOLD = 0.8        # Cumulative variance threshold for PCA (80%)
SIMILAR_CORR_THRESHOLD = 0.05   # Threshold for similar correlations
VARIABLES = [
    "pres_msl", "pres_sfc",
    "hgt_pres_925", "hgt_pres_850", "hgt_pres_700",
    "tmp_2m", "tmp_pres_925", "tmp_pres_850", "tmp_pres_700", "tmin_2m", "tmax_2m",
    "ugrd_hgt", "ugrd_pres_925", "ugrd_pres_850", "ugrd_pres_700",
    "vgrd_hgt", "vgrd_pres_925", "vgrd_pres_850", "vgrd_pres_700",
    "dswrf_sfc", "dlwrf_sfc", "uswrf_sfc", "ulwrf_sfc", "lhtfl_sfc", "shtfl_sfc",
    "soilw_bgrnd",
    "tcdc_eatm"
]
EXCLUDE_VARS_DICT = {
    "tmax_obs": ["tmin_2m", "tmp_2m"],
}

class StopWhenNoImprovement:
    """Custom callback for early stopping in Optuna when RMSE stops improving."""
    
    def __init__(self, patience=5):
        self.patience = patience
        self.best_value = float('inf')
        self.best_trial = 0
        self.n_trials_ran = 0

    def __call__(self, study, trial):
        """Check if optimization should stop based on RMSE improvement."""
        self.n_trials_ran += 1
        current_value = study.best_value
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_trial = self.n_trials_ran
        elif self.n_trials_ran - self.best_trial >= self.patience:
            study.stop()
            print(f"Stopping optimization early after {self.n_trials_ran} trials "
                  f"due to no improvement in RMSE for {self.patience} trials.")

def plot_error_histogram(actual, predicted, output_dir, base_filename, station_id, fhour, 
                        title_prefix, error_type='mae'):
    """Generate and save histogram of prediction errors (ME, MAE, or RMSE).

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.
        output_dir: Directory to save the plot.
        base_filename: Base filename for the plot.
        station_id: Station ID for plot title.
        fhour: Forecast hour for plot title.
        title_prefix: Prefix for plot title.
        error_type: Type of error to plot ('me', 'mae', or 'rmse').
    """
    if error_type == 'me':
        errors = predicted - actual
        mean_error = np.mean(errors)
        label = f'ME = {mean_error:.2f} °C'
        filename_suffix = '_me.png'
        xlabel = 'Error (°C)'
        title_suffix = 'ME'
    elif error_type == 'mae':
        errors = np.abs(predicted - actual)
        mean_error = mean_absolute_error(actual, predicted)
        label = f'MAE = {mean_error:.2f} °C'
        filename_suffix = '_mae.png'
        xlabel = 'Absolute Error (°C)'
        title_suffix = 'MAE'
    else:  # rmse
        errors = (predicted - actual) ** 2
        mean_error = root_mean_squared_error(actual, predicted)
        label = f'RMSE = {mean_error:.2f} °C'
        filename_suffix = '_rmse.png'
        xlabel = 'Squared Error (°C²)'
        title_suffix = 'RMSE'
    
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=30, kde=True, color='purple')
    plt.axvline(mean_error if error_type != 'rmse' else mean_error**2, 
                color='red', linestyle='--', label=label)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(f'{title_prefix} {title_suffix} Distribution\n'
              f'(Station: {station_id}, Forecast Hour: {fhour})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"{base_filename}{filename_suffix}")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"{title_suffix} histogram saved to {plot_file}")

def get_pressure_level(var_name):
    """Extract pressure level from variable name.

    Args:
        var_name: Variable name (e.g., 'tmp_pres_850').

    Returns:
        int or None: Pressure level if present, else None.
    """
    try:
        if '_pres_' in var_name:
            return int(var_name.split('_pres_')[-1])
        return None
    except:
        return None

def get_base_parameter(var_name):
    """Extract base parameter type from variable name.

    Args:
        var_name: Variable name (e.g., 'tmp_2m').

    Returns:
        str: Base parameter name.
    """
    if var_name == 'hgt_ceiling':
        return 'hgt_ceiling'
    if var_name == 'ugrd_hgt':
        return 'ugrd_hgt'
    if var_name == 'vgrd_hgt':
        return 'vgrd_hgt'
    if '_2m' in var_name:
        return var_name.split('_2m')[0]
    elif '_pres_' in var_name:
        return var_name.split('_pres_')[0]
    elif '_' in var_name:
        parts = var_name.rsplit('_', 1)
        return parts[0] if len(parts) > 1 else var_name
    return var_name

def quality_control_data(df, target_variable, variables, station_id, fhour, output_dir, source='GEFS'):
    """Apply quality control to raw data using IQR method and temporal consistency checks.

    Args:
        df: DataFrame with raw data.
        target_variable: Target variable (e.g., 'tmax_obs').
        variables: List of predictor variables.
        station_id: Station ID for context.
        fhour: Forecast hour for context.
        output_dir: Directory to save QC outputs.
        source: Data source ('GEFS' or ('NBM') for logging purposes.

    Returns:
        DataFrame or None: Cleaned DataFrame or None if no valid data remains.
    """
    print(f"Performing quality control for {source} station {station_id}, forecast hour {fhour}")

    # Initialize log and record initial row count
    initial_rows = len(df)
    qc_log = [f"Quality Control Log for {source} Station: {station_id}, Forecast Hour: {fhour}",
              f"Initial rows: {initial_rows}"]

    # Step 1: Detect outliers in target variable using IQR method
    Q1 = df[target_variable].quantile(0.25)
    Q3 = df[target_variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[target_variable].apply(lambda x: x < lower_bound or x > upper_bound)
    outlier_count = outliers.sum()
    if outlier_count > 0:
        print(f"Found {outlier_count} outliers in {target_variable} for {source} station {station_id}")
        # Visualize distribution with IQR bounds
        plt.figure(figsize=(8, 6))
        sns.histplot(df[target_variable], bins=30, kde=True, color='blue')
        plt.axvline(lower_bound, color='red', linestyle='--', 
                    label=f'Lower IQR Bound: {lower_bound:.2f}')
        plt.axvline(upper_bound, color='red', linestyle='--', 
                    label=f'Upper IQR Bound: {upper_bound:.2f}')
        plt.xlabel(target_variable)
        plt.ylabel('Count')
        plt.title(f'Distribution of {target_variable} with IQR Bounds\n'
                  f'({source} Station: {station_id}, Forecast Hour: {fhour})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"qc_distribution_{target_variable}_{source}_{station_id}_f{fhour:03d}.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"QC distribution plot for {target_variable} ({source}) saved to {plot_file}")
        # Remove outliers
        df = df[~outliers]
        print(f"Dropped {outlier_count} rows due to outliers in {target_variable} ({source})")
    qc_log.append(f"Rows dropped due to outliers in {target_variable}: {outlier_count}")

    if df.empty:
        print(f"No valid data after outlier removal for {source} station {station_id}, forecast hour {fhour}")
        qc_log.append(f"No valid data after outlier removal")
        with open(os.path.join(output_dir, f"qc_log_{source}_{station_id}_f{fhour:03d}.txt"), 'w') as f:
            f.write("\n".join(qc_log))
        print(f"QC log saved to {os.path.join(output_dir, f'qc_log_{source}_{station_id}_f{fhour:03d}.txt')}")
        return None

    # Step 2: Check temporal consistency for target variable
    df = df.sort_values(by='valid_datetime')
    tmax_diff = df[target_variable].diff().abs()
    max_diff_threshold = 15  # Maximum allowable change in tmax_obs (°C)
    large_jumps = tmax_diff > max_diff_threshold
    jump_count = large_jumps.sum()
    if jump_count > 0:
        print(f"Found {jump_count} large jumps in {target_variable} exceeding {max_diff_threshold}°C ({source})")
        # Visualize temporal differences
        plt.figure(figsize=(10, 6))
        plt.plot(df['valid_datetime'], tmax_diff, label=f'|Δ {target_variable}|')
        plt.axhline(max_diff_threshold, color='red', linestyle='--', 
                    label=f'Threshold: {max_diff_threshold}°C')
        plt.xlabel('Date')
        plt.ylabel(f'|Δ {target_variable}| (°C)')
        plt.title(f'Temporal Differences in {target_variable}\n'
                  f'({source} Station: {station_id}, Forecast Hour: {fhour})')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"qc_temporal_diff_{target_variable}_{source}_{station_id}_f{fhour:03d}.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"QC temporal difference plot for {target_variable} ({source}) saved to {plot_file}")
        # Remove rows with large jumps
        df = df[~large_jumps]
        print(f"Dropped {jump_count} rows due to large jumps in {target_variable} ({source})")
    qc_log.append(f"Rows dropped due to large jumps in {target_variable}: {jump_count}")

    if df.empty:
        print(f"No valid data after temporal consistency checks for {source} station {station_id}, forecast hour {fhour}")
        qc_log.append(f"No valid data after temporal consistency checks")
        with open(os.path.join(output_dir, f"qc_log_{source}_{station_id}_f{fhour:03d}.txt"), 'w') as f:
            f.write("\n".join(qc_log))
        print(f"QC log saved to {os.path.join(output_dir, f'qc_log_{source}_{station_id}_f{fhour:03d}.txt')}")
        return None

    # Step 3: Log and save final results
    final_rows = len(df)
    rows_removed = initial_rows - final_rows
    qc_log.append(f"Final rows: {final_rows}")
    qc_log.append(f"Total rows removed: {rows_removed}")
    print(f"Quality control completed for {source}: {rows_removed} rows removed, {final_rows} rows remain")
    
    with open(os.path.join(output_dir, f"qc_log_{source}_{station_id}_f{fhour:03d}.txt"), 'w') as f:
        f.write("\n".join(qc_log))
    print(f"QC log saved to {os.path.join(output_dir, f'qc_log_{source}_{station_id}_f{fhour:03d}.txt')}")

    return df

def extract_station_data_nbm(file_path, station_id, variables, target_variable, fhour, output_dir):
    """Extract and preprocess NBM data for a single station and a single forecast hour.

    Args:
        file_path: File path for NBM station data (single file for the station).
        station_id: Station ID (single ID, e.g., PROCESS_STATION).
        variables: List of predictor variables.
        target_variable: Target variable for prediction.
        fhour: Forecast hour.
        output_dir: Directory to save QC outputs.

    Returns:
        DataFrame or None: DataFrame for the station or None if no valid data.
    """
    try:
        if not os.path.exists(file_path):
            print(f"NBM input file {file_path} not found for station {station_id}.")
            return None
        df = pd.read_csv(file_path)
        df["valid_datetime"] = pd.to_datetime(df["valid_datetime"])
        df = df[df["sid"] == station_id]
        if df.empty:
            print(f"No data found for station {station_id} in {file_path} for NBM data.")
            return None
        # Apply quality control
        df = quality_control_data(df, target_variable, variables, station_id, fhour, output_dir, source='NBM')
        if df is None or df.empty:
            print(f"No valid data after quality control for NBM station {station_id}.")
            return None
        df["forecast_hour"] = fhour
        df["date"] = df["valid_datetime"]
        
        return df.copy()
    except Exception as e:
        print(f"Error processing NBM file for station {station_id}, forecast hour {fhour}: {e}")
        return None

def extract_station_data_gefs(file_path, station_id, start_date, end_date, variables, 
                              target_variable, fhour, output_dir):
    """Extract and preprocess GEFS data for a single station and a single forecast hour.

    Args:
        file_path: File path for GEFS station data (single file for the station).
        station_id: Station ID (single ID, e.g., PROCESS_STATION).
        start_date: Start date for data filtering.
        end_date: End date for data filtering.
        variables: List of predictor variables.
        target_variable: Target variable for prediction.
        fhour: Forecast hour.
        output_dir: Directory to save QC outputs.

    Returns:
        DataFrame or None: DataFrame for the station or None if no valid data.
    """
    try:
        if not os.path.exists(file_path):
            print(f"GEFS input file {file_path} not found for station {station_id}.")
            return None
        df = pd.read_csv(file_path)
        df["valid_datetime"] = pd.to_datetime(df["valid_datetime"])
        df = df[df["sid"] == station_id]
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        df = df[(df["valid_datetime"] >= start) & 
                (df["valid_datetime"] <= end + timedelta(days=1))]
        if df.empty:
            print(f"No data found for station {station_id} in {file_path} for the specified date range.")
            return None
        # Apply quality control
        df = quality_control_data(df, target_variable, variables, station_id, fhour, output_dir, source='GEFS')
        if df is None or df.empty:
            print(f"No valid data after quality control for GEFS station {station_id}.")
            return None
        df["forecast_hour"] = fhour
        df["date"] = df["valid_datetime"]
        
        cols = ["sid", "valid_datetime", "forecast_hour", "date", "perturbation"] + \
               variables + [target_variable]
        if not all(col in df.columns for col in cols):
            print(f"Missing columns in GEFS data for forecast hour {fhour}: "
                  f"{[col for col in cols if col not in df.columns]}")
            return None
        
        return df[cols].copy()
    except Exception as e:
        print(f"Error processing GEFS file for station {station_id}, forecast hour {fhour}: {e}")
        return None

def plot_four_panel_scatters(gefs_df, nbm_df, y_train, y_pred_train, y_test, y_pred_test, 
                             output_dir, fhour, station_id):
    """Generate a 4-panel scatter plot for GEFS, NBM, training, and testing data for the specified station.

    Args:
        gefs_df: DataFrame with GEFS data for the station (containing tmax_2m and tmax_obs).
        nbm_df: DataFrame with NBM data for the station (containing tmax_2m and tmax_obs).
        y_train: Actual target values for training set.
        y_pred_train: Predicted target values for training set.
        y_test: Actual target values for test set.
        y_pred_test: Predicted target values for test set.
        output_dir: Directory to save the plot.
        fhour: Forecast hour for plot title.
        station_id: Station ID for titles and file names.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Helper function to add metrics to a plot
    def add_metrics(ax, actual, predicted, title):
        point_count = len(actual)
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = root_mean_squared_error(actual, predicted)
        metrics_text = (f'Points: {point_count}\n'
                       f'R²: {r2:.4f}\n'
                       f'MAE: {mae:.2f} °C\n'
                       f'RMSE: {rmse:.2f} °C')
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.scatter(actual, predicted, alpha=0.5)
        plot_min = min(actual.min(), predicted.min())
        plot_max = max(actual.max(), predicted.max())
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2)
        ax.set_xlabel('Observed tmax_obs (°C)')
        ax.set_ylabel(f'{title} (°C)')
        ax.set_title(f'{title} vs Observed tmax_obs\n(Station: {station_id}, Forecast Hour: {fhour})')  # Modified: Dynamic station_id
        ax.grid(True)

    # Top Left: GEFS tmax_2m vs tmax_obs
    if 'tmax_2m' in gefs_df.columns and 'tmax_obs' in gefs_df.columns:
        plot_df_gefs = gefs_df[['tmax_2m', 'tmax_obs']].dropna()
        if not plot_df_gefs.empty:
            add_metrics(axes[0], plot_df_gefs['tmax_obs'], plot_df_gefs['tmax_2m'], 'GEFS tmax_2m')
        else:
            axes[0].text(0.5, 0.5, 'No valid GEFS data', ha='center', va='center')
            axes[0].set_xlabel('Observed tmax_obs (°C)')
            axes[0].set_ylabel('GEFS tmax_2m (°C)')
            axes[0].set_title(f'GEFS tmax_2m vs Observed tmax_obs\n(Station: {station_id}, Forecast Hour: {fhour})')  # Modified: Dynamic station_id
    else:
        axes[0].text(0.5, 0.5, 'GEFS columns missing', ha='center', va='center')
        axes[0].set_xlabel('Observed tmax_obs (°C)')
        axes[0].set_ylabel('GEFS tmax_2m (°C)')
        axes[0].set_title(f'GEFS tmax_2m vs Observed tmax_obs\n(Station: {station_id}, Forecast Hour: {fhour})')  # Modified: Dynamic station_id

    # Top Right: NBM tmax_2m vs tmax_obs
    if 'tmax_2m' in nbm_df.columns and 'tmax_obs' in nbm_df.columns:
        plot_df_nbm = nbm_df[['tmax_2m', 'tmax_obs']].dropna()
        if not plot_df_nbm.empty:
            add_metrics(axes[1], plot_df_nbm['tmax_obs'], plot_df_nbm['tmax_2m'], 'NBM tmax_2m')
        else:
            axes[1].text(0.5, 0.5, 'No valid NBM data', ha='center', va='center')
            axes[1].set_xlabel('Observed tmax_obs (°C)')
            axes[1].set_ylabel('NBM tmax_2m (°C)')
            axes[1].set_title(f'NBM tmax_2m vs Observed tmax_obs\n(Station: {station_id}, Forecast Hour: {fhour})')  # Modified: Dynamic station_id
    else:
        axes[1].text(0.5, 0.5, 'NBM columns missing', ha='center', va='center')
        axes[1].set_xlabel('Observed tmax_obs (°C)')
        axes[1].set_ylabel('NBM tmax_2m (°C)')
        axes[1].set_title(f'NBM tmax_2m vs Observed tmax_obs\n(Station: {station_id}, Forecast Hour: {fhour})')  # Modified: Dynamic station_id

    # Bottom Left: Training Set (Actual vs Predicted)
    add_metrics(axes[2], y_train, y_pred_train, 'Predicted tmax_obs (Training)')
    axes[2].set_title(f'Actual vs Predicted tmax_obs (Training)\n(Station: {station_id}, Forecast Hour: {fhour})')  # Modified: Dynamic station_id
    axes[2].set_ylabel('Predicted tmax_obs (°C)')

    # Bottom Right: Testing Set (Actual vs Predicted)
    add_metrics(axes[3], y_test, y_pred_test, 'Predicted tmax_obs (Testing)')
    axes[3].set_title(f'Actual vs Predicted tmax_obs (Testing)\n(Station: {station_id}, Forecast Hour: {fhour})')  # Modified: Dynamic station_id
    axes[3].set_ylabel('Predicted tmax_obs (°C)')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"four_panel_scatter_{station_id}_f{fhour:03d}.png")  # Modified: Dynamic station_id
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Four-panel scatter plot saved to {plot_file}")

def objective(trial, X_train, y_train, cv=5, random_state=42):
    """Optuna objective function for hyperparameter tuning of XGBoost.

    Args:
        trial: Optuna trial object.
        X_train: Training features.
        y_train: Training target.
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        float: Mean RMSE from cross-validation.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 10, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': random_state
    }
    
    model = XGBRegressor(**params)
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rmses = []
    
    for step, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred_val = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred_val)
        rmses.append(rmse)
        
        trial.report(rmse, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(rmses)

def main():
    """Main function to process GEFS and NBM weather data for the specified station, apply quality control, and train model."""
    # Load station metadata (for consistency, though only PROCESS_STATION is used)
    stations_df = pd.read_csv(STATIONS_FILE)
    station_id = PROCESS_STATION  # Modified: Use single station ID

    for fhour in FORECAST_HOURS:
        print(f"\nProcessing station: {station_id} for forecast hour: {fhour}")
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"f{fhour:03d}")
        os.makedirs(output_dir, exist_ok=True)

        # Modified: Prepare GEFS file path for the specified station
        gefs_file_path = os.path.join(GEFS_DATA_DIR, f"f{fhour:03d}", 
                                     f"{station_id}_{GEFS_START_DATE[:4]}_{GEFS_END_DATE[:4]}_f{fhour:03d}.csv")

        # Modified: Prepare NBM file path for the specified station
        nbm_file_path = os.path.join(NBM_DATA_DIR, f"f{fhour:03d}", 
                                    f"{station_id}_2023_2024_f{fhour:03d}.csv")

        print(f"Loading GEFS data from: {gefs_file_path}")
        gefs_df = extract_station_data_gefs(gefs_file_path, station_id, GEFS_START_DATE, 
                                            GEFS_END_DATE, VARIABLES, TARGET_VARIABLE, fhour, output_dir)

        print(f"Loading NBM data from: {nbm_file_path}")
        nbm_df = extract_station_data_nbm(nbm_file_path, station_id, VARIABLES, 
                                         TARGET_VARIABLE, fhour, output_dir)

        if gefs_df is None or gefs_df.empty:
            print(f"No GEFS data collected for station {station_id} at forecast hour {fhour}. "
                  f"Continuing to next forecast hour.")
            continue

        # Drop remaining NaN values for GEFS data (used for modeling)
        gefs_df = gefs_df.dropna()
        if gefs_df.empty:
            print(f"No valid GEFS data after dropping NaNs for station {station_id} at forecast hour {fhour}.")
            continue

        # Save station and forecast hour info
        with open(os.path.join(output_dir, f"station_f{fhour:03d}.txt"), "w") as f:
            f.write(f"Station: {station_id}\nForecast Hour: {fhour}")
        print(f"Station and forecast hour info saved to station_f{fhour:03d}.txt")

        # Correlation and feature selection (using GEFS data)
        correlation_matrix = gefs_df[VARIABLES + [TARGET_VARIABLE]].corr(method="pearson")
        tmax_correlations = correlation_matrix[TARGET_VARIABLE].drop(TARGET_VARIABLE)

        # Step 1: Exclude manually specified variables
        exclude_vars = EXCLUDE_VARS_DICT.get(TARGET_VARIABLE, [])
        if exclude_vars:
            print(f"\nExcluding manually specified variables for target {TARGET_VARIABLE} "
                  f"for station {station_id}: {exclude_vars}")  # Modified: Dynamic station_id
        else:
            print(f"\nNo manually specified variables to exclude for target {TARGET_VARIABLE} "
                  f"for station {station_id}")  # Modified: Dynamic station_id

        feature_cols = [var for var in VARIABLES if var not in exclude_vars]
        print(f"\nFeatures after excluding manually specified variables: {feature_cols}")

        # Step 2: Select one variation per parameter type based on correlation
        param_groups = {}
        for var in feature_cols:
            base_param = get_base_parameter(var)
            param_groups.setdefault(base_param, []).append(var)

        selected_vars = []
        excluded_vars = set(exclude_vars)
        for base_param, var_list in param_groups.items():
            if len(var_list) == 1:
                selected_vars.append(var_list[0])
                print(f"Keeping {var_list[0]} for {base_param} (only variation)")
            else:
                valid_vars = [var for var in var_list if var not in exclude_vars]
                if not valid_vars:
                    print(f"Skipping {base_param} (all variations excluded by dictionary)")
                    continue
                correlations = [(var, abs(tmax_correlations[var])) for var in valid_vars]
                correlations.sort(key=lambda x: x[1], reverse=True)
                max_corr = correlations[0][1]
                candidates = [var for var, corr in correlations 
                              if max_corr - corr <= SIMILAR_CORR_THRESHOLD]
                
                if len(candidates) == 1:
                    selected_var = candidates[0]
                    selected_vars.append(selected_var)
                    print(f"Selected {selected_var} for {base_param} with highest target "
                          f"correlation ({tmax_correlations[selected_var]:.4f}) among {valid_vars}")
                else:
                    candidate_levels = [(var, get_pressure_level(var)) for var in candidates]
                    candidate_levels = [(var, level if level is not None else 10000) 
                                        for var, level in candidate_levels]
                    candidate_levels.sort(key=lambda x: x[1])
                    selected_var = candidate_levels[0][0]
                    selected_vars.append(selected_var)
                    print(f"Selected {selected_var} for {base_param} with correlation "
                          f"{tmax_correlations[selected_var]:.4f} "
                          f"(level: {get_pressure_level(selected_var)}) among {valid_vars} "
                          f"(similar correlations)")
                
                excluded_vars.update(set(var_list) - {selected_var})

        feature_cols = selected_vars
        print(f"\nFeatures after selecting one variation per parameter type: {feature_cols}")

        # Step 3: Exclude variables highly correlated with the target
        high_corr_target_vars = tmax_correlations[feature_cols][
            abs(tmax_correlations[feature_cols]) >= TARGET_CORR_THRESHOLD].index.tolist()
        if high_corr_target_vars:
            print(f"\nExcluding variables with |correlation| >= {TARGET_CORR_THRESHOLD} "
                  f"with {TARGET_VARIABLE} for station {station_id}: {high_corr_target_vars}")  # Modified: Dynamic station_id
            excluded_vars.update(high_corr_target_vars)
        else:
            print(f"\nNo variables with |correlation| >= {TARGET_CORR_THRESHOLD} "
                  f"with {TARGET_VARIABLE} for station {station_id}")  # Modified: Dynamic station_id

        feature_cols = [var for var in feature_cols if var not in high_corr_target_vars]
        print(f"\nFinal features retained for PCA: {feature_cols}")

        # Save selected feature columns
        with open(os.path.join(output_dir, f"feature_columns_{station_id}_f{fhour:03d}.txt"), "w") as f:  # Modified: Dynamic station_id
            f.write("\n".join(feature_cols))
        print(f"Selected feature columns saved to feature_columns_{station_id}_f{fhour:03d}.txt")

        # Save excluded variables
        all_excluded_vars = list(excluded_vars)
        with open(os.path.join(output_dir, f"excluded_variables_{station_id}_f{fhour:03d}.txt"), "w") as f:  # Modified: Dynamic station_id
            f.write("\n".join(all_excluded_vars) if all_excluded_vars else "None")
        print(f"Excluded variables saved to excluded_variables_{station_id}_f{fhour:03d}.txt")

        # Save and print correlation matrix
        print(f"\nFull Correlation Matrix for station {station_id} (rounded to 2 decimals):")  # Modified: Dynamic station_id
        print(correlation_matrix.round(2))
        correlation_matrix.to_csv(os.path.join(output_dir, 
                                              f"correlation_matrix_{station_id}_f{fhour:03d}.csv"))  # Modified: Dynamic station_id
        print(f"Correlation matrix saved to correlation_matrix_{station_id}_f{fhour:03d}.csv")

        correlation_pairs = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        correlation_pairs = correlation_pairs.stack().reset_index()
        correlation_pairs.columns = ["Variable_1", "Variable_2", "Correlation"]
        correlation_pairs = correlation_pairs.dropna()
        correlation_pairs = correlation_pairs.sort_values(by="Correlation", key=abs, ascending=False)
        correlation_pairs.to_csv(os.path.join(output_dir, 
                                             f"correlation_pairs_{station_id}_f{fhour:03d}.csv"), 
                                 index=False)  # Modified: Dynamic station_id
        print(f"Pairwise correlations saved to correlation_pairs_{station_id}_f{fhour:03d}.csv")

        tmax_correlations.to_csv(os.path.join(output_dir, 
                                             f"tmax_correlations_{station_id}_f{fhour:03d}.csv"))  # Modified: Dynamic station_id
        print(f"Correlations with tmax_obs saved to tmax_correlations_{station_id}_f{fhour:03d}.csv")

        # Generate correlation matrix heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1, 
                    annot=True, fmt=".2f")
        plt.title(f"Correlation Matrix of All Variables (Station: {station_id}, Forecast Hour: {fhour})")  # Modified: Dynamic station_id
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"correlation_matrix_heatmap_{station_id}_f{fhour:03d}.png"), 
                    dpi=300)  # Modified: Dynamic station_id
        plt.close()
        print(f"Correlation matrix heatmap saved to correlation_matrix_heatmap_{station_id}_f{fhour:03d}.png")

        top_n = 10
        print(f"\nTop {top_n} pairwise correlations for station {station_id} (absolute value):")  # Modified: Dynamic station_id
        print(correlation_pairs.head(top_n))

        print(f"\nTop {top_n} variables most correlated with tmax_obs for station {station_id} (absolute value):")  # Modified: Dynamic station_id
        top_tmax_correlations = tmax_correlations.abs().sort_values(ascending=False).head(top_n)
        print(top_tmax_correlations)

        if not feature_cols:
            print(f"No features remaining for PCA after filtering for station {station_id}. Continuing.")  # Modified: Dynamic station_id
            continue

        # Apply PCA
        X = gefs_df[feature_cols].values
        y = gefs_df[TARGET_VARIABLE].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, os.path.join(output_dir, f"scaler_{station_id}_f{fhour:03d}.pkl"))  # Modified: Dynamic station_id
        print(f"Saved StandardScaler to scaler_{station_id}_f{fhour:03d}.pkl")

        n_components = min(len(feature_cols), X_scaled.shape[0])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        joblib.dump(pca, os.path.join(output_dir, f"pca_{station_id}_f{fhour:03d}.pkl"))  # Modified: Dynamic station_id
        print(f"Saved PCA to pca_{station_id}_f{fhour:03d}.pkl")

        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        pca_results = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(n_components)],
            "Explained_Variance_Ratio": explained_variance_ratio,
            "Cumulative_Variance": cumulative_variance
        })

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=feature_cols
        )

        pca_results.to_csv(os.path.join(output_dir, f"pca_variance_{station_id}_f{fhour:03d}.csv"), 
                           index=False)  # Modified: Dynamic station_id
        loadings.to_csv(os.path.join(output_dir, f"pca_loadings_{station_id}_f{fhour:03d}.csv"))  # Modified: Dynamic station_id
        df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
        df_pca["tmax_obs"] = y
        df_pca["sid"] = gefs_df["sid"]
        df_pca["date"] = gefs_df["date"]
        df_pca["forecast_hour"] = gefs_df["forecast_hour"]
        df_pca["perturbation"] = gefs_df["perturbation"]
        df_pca.to_csv(os.path.join(output_dir, f"pca_transformed_data_{station_id}_f{fhour:03d}.csv"), 
                      index=False)  # Modified: Dynamic station_id
        print(f"PCA transformed data saved to pca_transformed_data_{station_id}_f{fhour:03d}.csv")

        # Plot PCA variance
        num_pcs_to_plot = min(15, n_components)
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, num_pcs_to_plot + 1), explained_variance_ratio[:num_pcs_to_plot], 
                alpha=0.5, align='center', label='Individual')
        plt.plot(range(1, num_pcs_to_plot + 1), cumulative_variance[:num_pcs_to_plot], 
                 marker='o', color='r', label='Cumulative')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title(f'Variance Explained by First {num_pcs_to_plot} PCs for Station {station_id}')  # Modified: Dynamic station_id
        plt.xticks(range(1, num_pcs_to_plot + 1))
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'variance_explained_{station_id}_f{fhour:03d}.png'), 
                    dpi=300)  # Modified: Dynamic station_id
        plt.close()
        print(f"Variance explained plot saved to variance_explained_{station_id}_f{fhour:03d}.png")

        # Analyze PCA correlations
        correlations = df_pca.corr(numeric_only=True)["tmax_obs"].drop("tmax_obs")
        print(f"\nCorrelation of PCs with tmax_obs for station {station_id}:")  # Modified: Dynamic station_id
        print(correlations)

        best_pc = correlations.abs().idxmax()
        best_corr = correlations[best_pc]
        print(f"\nBest PC for tmax_obs correlation: {best_pc} (Correlation: {best_corr:.4f})")

        pcs_to_plot = np.where(cumulative_variance >= VARIANCE_THRESHOLD)[0]
        if len(pcs_to_plot) > 0:
            pcs_to_plot = pcs_to_plot[0] + 1
        else:
            pcs_to_plot = n_components
        print(f"Generating bar plots for {pcs_to_plot} PCs to explain at least "
              f"{VARIANCE_THRESHOLD*100}% variance (Cumulative Variance: "
              f"{cumulative_variance[pcs_to_plot-1]:.4f})")

        # Generate PCA loading plots
        for i in range(pcs_to_plot):
            pc = f"PC{i+1}"
            plt.figure(figsize=(14, 6))
            pc_loadings = loadings[pc]
            pc_loadings = pc_loadings[abs(pc_loadings) > LOADING_THRESHOLD]
            if not pc_loadings.empty:
                pc_loadings = pc_loadings.sort_values(key=abs, ascending=False)
                pc_loadings.plot(kind="bar")
                plt.title(f"PCA Loadings for {pc} (|Loading| > {LOADING_THRESHOLD}) "
                          f"(Station: {station_id}, Forecast Hour: {fhour})")  # Modified: Dynamic station_id
                plt.xlabel("Variables")
                plt.ylabel("Loading Value")
                plt.xticks(rotation=45, ha="right")
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"pca_loadings_bar_{pc}_{station_id}_f{fhour:03d}.png"), 
                            dpi=300)  # Modified: Dynamic station_id
                plt.close()
                print(f"Bar plot for {pc} loadings saved to pca_loadings_bar_{pc}_{station_id}_f{fhour:03d}.png")
                top_n_loadings = 10
                top_loadings = pc_loadings.abs().sort_values(ascending=False).head(top_n_loadings)
                print(f"\nTop {top_n_loadings} variables contributing to {pc} for station {station_id} "
                      f"(|loading| > {LOADING_THRESHOLD}):")  # Modified: Dynamic station_id
                print(top_loadings)
            else:
                print(f"No variables with |loading| > {LOADING_THRESHOLD} for {pc} in station {station_id}. "
                      f"Skipping bar plot.")  # Modified: Dynamic station_id

        print(f"\nPCA Results for station {station_id} (using {len(feature_cols)} features):")  # Modified: Dynamic station_id
        print(pca_results)
        print(f"\nComponent Loadings for station {station_id}:")  # Modified: Dynamic station_id
        print(loadings)

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.3, random_state=42
        )
        print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

        # Optimize XGBoost hyperparameters
        study = optuna.create_study(direction='minimize', pruner=MedianPruner(n_warmup_steps=5))
        early_stopping = StopWhenNoImprovement(patience=5)
        start_time = time.time()
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, cv=5, random_state=42),
            callbacks=[early_stopping],
            n_trials=30
        )
        optuna_time = time.time() - start_time
        n_trials_ran = early_stopping.n_trials_ran
        print(f"Optuna optimization took {optuna_time:.2f} seconds for {n_trials_ran} trials")

        # Train final model with best parameters
        best_params = study.best_params
        best_cv_rmse = study.best_value
        best_model = XGBRegressor(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        # Predict on training and test sets
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # Calculate performance metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_train = root_mean_squared_error(y_train, y_pred_train)
        rmse_test = root_mean_squared_error(y_test, y_pred_test)

        print(f"\nXGBoost with Optuna Tuning for station {station_id}, Forecast Hour {fhour}:")  # Modified: Dynamic station_id
        print(f"Training R² Score: {r2_train:.4f}")
        print(f"Testing R² Score: {r2_test:.4f}")
        print(f"Training MAE: {mae_train:.4f} °C")
        print(f"Testing MAE: {mae_test:.4f} °C")
        print(f"Training RMSE: {rmse_train:.4f} °C")
        print(f"Testing RMSE: {rmse_test:.4f} °C")
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validated RMSE score: {best_cv_rmse:.4f}")

        # Save the best model
        model_file = os.path.join(output_dir, f"xgb_model_{station_id}_f{fhour:03d}.pkl")  # Modified: Dynamic station_id
        joblib.dump(best_model, model_file)
        print(f"Saved best XGBoost model to {model_file}")

        # Save model feature columns (PCs)
        feature_columns = [f"PC{i+1}" for i in range(X_train.shape[1])]
        columns_file = os.path.join(output_dir, f"model_feature_columns_{station_id}_f{fhour:03d}.txt")  # Modified: Dynamic station_id
        with open(columns_file, "w") as f:
            f.write("\n".join(feature_columns))
        print(f"Model feature columns saved to {columns_file}")

        # Save regression results
        results_file = os.path.join(output_dir, f"regression_results_xgb_{station_id}_f{fhour:03d}.txt")  # Modified: Dynamic station_id
        with open(results_file, "w") as f:
            f.write(f"Station: {station_id}\n")
            f.write(f"Forecast Hour: {fhour}\n")
            f.write(f"Training R² Score for predicting {TARGET_VARIABLE}: {r2_train:.4f}\n")
            f.write(f"Testing R² Score for predicting {TARGET_VARIABLE}: {r2_test:.4f}\n")
            f.write(f"Training MAE for predicting {TARGET_VARIABLE}: {mae_train:.4f} °C\n")
            f.write(f"Testing MAE for predicting {TARGET_VARIABLE}: {mae_test:.4f} °C\n")
            f.write(f"Training RMSE for predicting {TARGET_VARIABLE}: {rmse_train:.4f} °C\n")
            f.write(f"Testing RMSE for predicting {TARGET_VARIABLE}: {rmse_test:.4f} °C\n")
            f.write(f"Optimization time (seconds): {optuna_time:.2f}\n")
            f.write(f"Number of trials run: {n_trials_ran}\n")
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best cross-validated RMSE score: {best_cv_rmse:.4f}\n")
            f.write("Feature importances for PCs:\n")
            for i, importance in enumerate(best_model.feature_importances_, 1):
                f.write(f"PC{i}: {importance:.4f}\n")
        print(f"Regression results saved to {results_file}")

        # Generate four-panel scatter plot
        plot_four_panel_scatters(gefs_df, nbm_df if nbm_df is not None else pd.DataFrame(),
                                 y_train, y_pred_train, y_test, y_pred_test,
                                 output_dir, fhour, station_id)  # Modified: Pass station_id

        # Generate error histograms for training and testing sets
        for error_type in ['mae', 'rmse']:
            plot_error_histogram(
                actual=y_train,
                predicted=y_pred_train,
                output_dir=output_dir,
                base_filename=f"regression_scatter_xgb_train_{station_id}_f{fhour:03d}",  # Modified: Dynamic station_id
                station_id=station_id,
                fhour=fhour,
                title_prefix=f"XGBoost Actual vs Predicted {TARGET_VARIABLE} (Training Set)",
                error_type=error_type
            )
            plot_error_histogram(
                actual=y_test,
                predicted=y_pred_test,
                output_dir=output_dir,
                base_filename=f"regression_scatter_xgb_test_{station_id}_f{fhour:03d}",  # Modified: Dynamic station_id
                station_id=station_id,
                fhour=fhour,
                title_prefix=f"XGBoost Actual vs Predicted {TARGET_VARIABLE} (Test Set)",
                error_type=error_type
            )

        # Generate error histograms for GEFS and NBM
        if 'tmax_2m' in gefs_df.columns and 'tmax_obs' in gefs_df.columns:
            plot_df_gefs = gefs_df[['tmax_2m', 'tmax_obs']].dropna()
            if not plot_df_gefs.empty:
                for error_type in ['me', 'rmse']:
                    plot_error_histogram(
                        actual=plot_df_gefs['tmax_obs'],
                        predicted=plot_df_gefs['tmax_2m'],
                        output_dir=output_dir,
                        base_filename=f"tmax_2m_vs_tmax_obs_{station_id}_gefs_f{fhour:03d}",  # Modified: Dynamic station_id
                        station_id=station_id,
                        fhour=fhour,
                        title_prefix="GEFS tmax_2m vs Observed tmax_obs",
                        error_type=error_type
                    )
        if nbm_df is not None and 'tmax_2m' in nbm_df.columns and 'tmax_obs' in nbm_df.columns:
            plot_df_nbm = nbm_df[['tmax_2m', 'tmax_obs']].dropna()
            if not plot_df_nbm.empty:
                for error_type in ['me', 'rmse']:
                    plot_error_histogram(
                        actual=plot_df_nbm['tmax_obs'],
                        predicted=plot_df_nbm['tmax_2m'],
                        output_dir=output_dir,
                        base_filename=f"tmax_2m_vs_tmax_obs_{station_id}_nbm_f{fhour:03d}",  # Modified: Dynamic station_id
                        station_id=station_id,
                        fhour=fhour,
                        title_prefix="NBM tmax_2m vs Observed tmax_obs",
                        error_type=error_type
                    )

if __name__ == "__main__":
    main()