#!/home/chad.kahler/anaconda3/envs/gefsai/bin/python
"""
GEFS Apply Model Site Specific Script

This script applies machine learning models to GEFS forecast data for site-specific predictions.
It processes forecast data, applies trained XGBoost models, and generates percentile forecasts.

Converted from Jupyter notebook to standalone Python script.
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
import joblib
import json
import sys
import os
import glob
import xgboost as xgb
import csv
import pickle
import time
import urllib.request
import urllib.parse
import traceback

def collectMaxMinData(dt, sid, token):
    params = urllib.parse.urlencode({
        "token": token,
        "start": f"{dt:%Y%m%d}",
        "end": f"{dt:%Y%m%d}",
        "period": "day",
        "obtimezone": "utc",
        "vars": "air_temp",
        "stid": sid,
        "statistic": "max,min",
        "units": "metric"
    })
    base_url = f"https://api.synopticdata.com/v2/stations/statistics?{params}"
    print(f"Collecting obs for: {dt:%b %d, %Y}")
    response = urllib.request.urlopen(base_url)
    return json.loads(response.read())

def collect_and_save_observations(model_datetime, output_dir, token, states, station_metadata_csv):
    try:
        station_output = os.path.join(output_dir, f"obs.{model_datetime:%Y%m%d}.csv")
        if not os.path.exists(station_output):
            df = pd.read_csv(station_metadata_csv)
            all_data = []
            for state in states:
                state_df = df[df['state'] == state]
                sids = state_df['sid'].unique()
                sids_string = ",".join(sids)
                maxmin_data = collectMaxMinData(model_datetime, sids_string, token)
                if maxmin_data and 'STATION' in maxmin_data:
                    for station in maxmin_data['STATION']:
                        if "max" in station["STATISTICS"]['air_temp_set_1'][0].keys():
                            tmax = station["STATISTICS"]['air_temp_set_1'][0]['max']
                            tmin = station["STATISTICS"]['air_temp_set_1'][0]['min']
                            station_data = {
                                'sid': station['STID'],
                                'lat': float(station['LATITUDE']),
                                'lon': float(station['LONGITUDE']),
                                'tmax_c': float(tmax) if tmax is not None else None,
                                'tmin_c': float(tmin) if tmin is not None else None,
                            }
                            all_data.append(station_data)
            all_df = pd.DataFrame(all_data, columns=['sid','lat','lon','tmax_c','tmin_c'])
            all_df.to_csv(station_output, index=False)
            print(f"Max/Min data saved for {len(all_df)} stations: {station_output}")
        else:
            print(f"Max/Min data already exists: {station_output}")
    except Exception as err:
        print(traceback.format_exc())


def main():
    """Main function to execute the GEFS model application workflow."""
    
    # Start timing
    start_time = time.time()

    ### SPECIFY MODEL DATETIME TO APPLY MODEL
    model_datetime = datetime(2025, 8, 1, 0)  # Year, Month, Day, Hour
    
    # Configuration parameters
    gefs_fcst_hour = [24, 48, 72, 96, 120, 144, 168]
    nbm_fcst_hour = [30, 54, 78, 102, 126, 150, 174]
    fcst_day = [1, 2, 3, 4, 5, 6, 7]
    
    print(f"Processing forecast for: {model_datetime}")
    print(f"XGBoost version: {xgb.__version__}")
    
    # Initialize variables
    all_percentiles = []
    remove_nans = True  # Set to True to remove NaNs from X and X_meta
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    # Download obs for each valid date in forecast period (separate process)
    obs_token = "294d54c2bfc9472d9f52f9816ae4c13c"
    obs_states = ["MT","ID","WA","OR","UT","NV","CA","AZ"]
    obs_station_metadata_csv = r"input/station_metadata.csv"
    obs_output_dir = os.path.join("input", f"{model_datetime:%Y%m%d}")
    # For each forecast hour, collect obs for model_datetime + forecast_hour - 24 hours
    valid_dates = [model_datetime + timedelta(hours=fh-24) for fh in gefs_fcst_hour]
    for valid_dt in valid_dates:
        collect_and_save_observations(valid_dt, obs_output_dir, obs_token, obs_states, obs_station_metadata_csv)

    # Model application and forecast/statistical logic remains unchanged
    for idx in range(7):
        print(f"\nProcessing forecast hour {gefs_fcst_hour[idx]}...")
        gefs_forecast_csv = fr"input/{model_datetime:%Y%m%d}/gefs_t00z_{model_datetime:%Y%m%d}_f{gefs_fcst_hour[idx]:03d}.csv"
        gefs_df = pd.read_csv(gefs_forecast_csv)

        nbm_forecast_csv = fr"input/{model_datetime:%Y%m%d}/nbm_t00z_{model_datetime:%Y%m%d}_f{nbm_fcst_hour[idx]:03d}.csv"
        nbm_df = pd.read_csv(nbm_forecast_csv)

        nbm_urma_csv = fr"input/{model_datetime:%Y%m%d}/nbm_urma_t00z_{model_datetime:%Y%m%d}_f{gefs_fcst_hour[idx]:03d}.csv"
        nbm_urma_df = pd.read_csv(nbm_urma_csv)
        nbm_urma_df.rename(columns={'nbm_tmax': 'nbm_det', 'urma_tmax': 'urma'}, inplace=True)

        gefs_df['valid_datetime'] = pd.to_datetime(gefs_df['valid_datetime'])
        gefs_df['month'] = gefs_df['valid_datetime'].dt.month
        gefs_df['day_of_year'] = gefs_df['valid_datetime'].dt.dayofyear

        gefs_df['gefs_lat'] = gefs_df['lat']
        gefs_df['gefs_lon'] = gefs_df['lon']
        gefs_df['gefs_elev'] = gefs_df['elev_ft']
        gefs_df['gefs_fcst_hour'] = gefs_df['fcst_hour']

        gefs_df['ensemble_mean'] = gefs_df['gefs_tmax_2m']
        gefs_df['ensemble_weighted'] = gefs_df['gefs_tmax_2m']
        gefs_df['solar_cloud_interaction'] = gefs_df['gefs_dswrf_sfc'] * (1 - gefs_df['gefs_tcdc_eatm'] / 100)
        gefs_df['temp_gradient_sfc_850'] = gefs_df['gefs_tmp_2m'] - gefs_df['gefs_tmp_pres_850']

        for sid in gefs_df['sid'].unique():
            print(f"Processing station: {sid}")
            site_df = gefs_df[gefs_df['sid'] == sid].reset_index(drop=True)
            site_model_files = glob.glob(fr"model/gefs_ml_tmax_{sid}_fhr24_*_latest.joblib")
            site_metadata_files = glob.glob(fr"model/gefs_ml_tmax_{sid}_fhr24_*_latest.json")
            if site_model_files and site_metadata_files:
                model_file_to_use = site_model_files[0]
                metadata_file_to_use = site_metadata_files[0]
                print(f"Using site-specific model for {sid}: {model_file_to_use}")
            else:
                # model_file_to_use = r"model/gefs_ml_tmax_KSLC_fhr24_gradient_boosting_latest.joblib"
                # metadata_file_to_use = r"model/gefs_ml_tmax_KSLC_fhr24_gradient_boosting_latest.json"
                
                model_file_to_use = r"model/gefs_ml_model_tmax_latest.joblib"
                metadata_file_to_use = r"model/gefs_ml_metadata_tmax_latest.json"

                print(f"No site-specific model found for {sid}, using KSLC model")
            try:
                print(f"Attempting to load model: {model_file_to_use}")
                trained_model = joblib.load(model_file_to_use)
                print(f"Successfully loaded model for {sid}")
            except Exception as e:
                print(f"Error loading model for {sid}: {e}")
                print(f"Model file: {model_file_to_use}")
                try:
                    print("Trying to load with pickle protocol...")
                    with open(model_file_to_use, 'rb') as f:
                        trained_model = pickle.load(f)
                    print("Successfully loaded with pickle")
                except Exception as e2:
                    print(f"Pickle loading also failed: {e2}")
                    print(f"Skipping station {sid} due to model loading error")
                    continue
            try:
                with open(metadata_file_to_use, 'r') as f:
                    model_metadata = json.load(f)
                selected_features = model_metadata['features']['selected_features']
            except Exception as e:
                print(f"Error loading metadata for {sid}: {e}")
                print(f"Metadata file: {metadata_file_to_use}")
                continue
            X = site_df[selected_features]
            X = X.loc[:, selected_features]
            X_meta = pd.DataFrame(site_df[['valid_datetime','sid','elev_ft','state','lat','lon','init_datetime','fcst_hour','perturbation','name']])
            if remove_nans:
                X_clean = X.dropna()
                valid_idx = X_clean.index
                X = X_clean.reset_index(drop=True)
                X_meta = X_meta.loc[valid_idx].reset_index(drop=True)
                gefs_tmax_2m_clean = site_df.loc[valid_idx, 'gefs_tmax_2m'].reset_index(drop=True)
            else:
                gefs_tmax_2m_clean = site_df['gefs_tmax_2m'].reset_index(drop=True)
            if len(X) == 0:
                print(f"Warning: No valid data for {sid} after removing NaNs. Skipping this station.")
                continue
            try:
                y = trained_model.predict(X)
                print(f"Successfully made predictions for {sid}: {len(y)} predictions")
            except Exception as e:
                print(f"Error making predictions for {sid}: {e}")
                continue
            if len(y) == 0:
                print(f"Warning: No predictions generated for {sid}. Skipping this station.")
                continue
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y = pd.DataFrame(y, index=X.index, columns=['gefs_tmax_2m_ml'])
            result = pd.concat([X_meta, y], axis=1)
            result['gefs_tmax_2m'] = gefs_tmax_2m_clean
            result['fcst_hour'] = gefs_fcst_hour[idx]
            if len(result) > 0:
                for p in percentiles:
                    result[f'gefs_tmax_2m_ml_p{p:02d}'] = result.groupby('sid')['gefs_tmax_2m_ml'].transform(lambda x: round(x.quantile(p / 100), 2))
                    result[f'gefs_tmax_2m_raw_p{p:02d}'] = result.groupby('sid')['gefs_tmax_2m'].transform(lambda x: round(x.quantile(p / 100), 2))
                X_mean = X.groupby(X_meta['sid']).mean().reset_index()
                try:
                    y_mean = trained_model.predict(X_mean.drop('sid', axis=1) if 'sid' in X_mean.columns else X_mean)
                    y_mean_df = pd.DataFrame({
                        'sid': X_mean['sid'] if 'sid' in X_mean.columns else X_meta.groupby('sid').first().reset_index()['sid'],
                        'gefs_mean_ml': [round(val, 2) for val in y_mean.flatten()]
                    })
                except Exception as e:
                    print(f"Error calculating ensemble mean prediction for {sid}: {e}")
                    y_mean_df = pd.DataFrame({
                        'sid': [sid],
                        'gefs_mean_ml': [None]
                    })
                stations_percentiles = result.loc[:, ['sid', 'name', 'elev_ft', 'state', 'lat', 'lon', 'fcst_hour'] +
                    [col for col in result.columns if col.startswith('gefs_tmax_2m_ml_p') or col.startswith('gefs_tmax_2m_raw_p')]]
                nbm_percentile_cols = [f'nbm_tmp_2m_p{p:02d}' for p in percentiles]
                nbm_subset = nbm_df[nbm_df['sid'] == sid][['sid'] + nbm_percentile_cols]
                nbm_urma_cols = ['nbm_det', 'urma']
                nbm_urma_subset = nbm_urma_df[nbm_urma_df['sid'] == sid][['sid'] + nbm_urma_cols]
                stations_percentiles = stations_percentiles.merge(nbm_subset, on='sid', how='left')
                stations_percentiles = stations_percentiles.merge(nbm_urma_subset, on='sid', how='left')
                stations_percentiles = stations_percentiles.merge(y_mean_df, on='sid', how='left')
                stations_percentiles.reset_index(drop=True, inplace=True)
                all_percentiles.append(stations_percentiles)
            else:
                print(f"Warning: No valid results for {sid}. Skipping this station.")
    
    # Concatenate all forecast hours into one DataFrame and save to a single file
    if all_percentiles:
        all_percentiles_df = pd.concat(all_percentiles, ignore_index=True)
        output_path = os.path.join("output", f"{model_datetime:%Y%m%d}")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        output_csv = fr"{output_path}/models_t00z_{model_datetime:%Y%m%d}_ml_percentiles.csv"
        all_percentiles_df.to_csv(output_csv, index=False)
        print(f"Successfully saved results for {len(all_percentiles_df)} records to {output_csv}")
        
        # Convert CSV to JSON format
        convert_to_json(model_datetime)
    else:
        print("Warning: No valid data to save.")
    
    # Calculate and print total execution time
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\nTotal execution time: {execution_time_minutes:.2f} minutes")


def convert_to_json(model_datetime):
    """
    Convert the CSV output to JSON format for easier consumption by other applications.
    
    Args:
        model_datetime (datetime): The model datetime for file naming
    """
    print("\nConverting CSV to JSON format...")

    csv_file = fr"output/{model_datetime:%Y%m%d}/models_t00z_{model_datetime:%Y%m%d}_ml_percentiles.csv"
    json_file = fr"output/{model_datetime:%Y%m%d}/models_t00z_{model_datetime:%Y%m%d}_ml_percentiles.json"

    # Define percentiles and models based on your columns
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    model_columns = {
        "gefs_ml": [f"gefs_tmax_2m_ml_p{str(p).zfill(2)}" for p in percentiles],
        "gefs_raw": [f"gefs_tmax_2m_raw_p{str(p).zfill(2)}" for p in percentiles],
        "nbm": [f"nbm_tmp_2m_p{str(p).zfill(2)}" for p in percentiles]
    }

    # Add nbm_det_tmax_2m, urma_tmax_2m, and gefs_mean_ml to each station/fcst_hour entry
    extra_cols = ["nbm_det", "urma", "gefs_mean_ml"]
    for model in extra_cols:
        model_columns[model] = [model]

    # Build obs_dicts for each valid date in the forecast period
    gefs_fcst_hour = [24, 48, 72, 96, 120, 144, 168]
    # For each forecast hour, obs is for model_datetime + forecast_hour - 24 hours
    valid_dates = [model_datetime + timedelta(hours=fh-24) for fh in gefs_fcst_hour]
    obs_dicts = {}
    for valid_dt in valid_dates:
        obs_file = fr"input/{model_datetime:%Y%m%d}/obs.{valid_dt:%Y%m%d}.csv"
        obs_dict = {}
        if os.path.exists(obs_file):
            try:
                with open(obs_file, newline='') as f_obs:
                    obs_reader = csv.DictReader(f_obs)
                    for row in obs_reader:
                        obs_dict[row['sid']] = row['tmax_c']
                print(f"Loaded observations from {obs_file}")
            except Exception as e:
                print(f"Warning: Could not load observations file {obs_file}: {e}")
        else:
            print(f"Warning: Observations file {obs_file} not found")
        obs_dicts[valid_dt.strftime('%Y%m%d')] = obs_dict

    output = {
        "percentiles": percentiles,
        "stations": {}
    }

    try:
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sid"]
                fcst_hour = row["fcst_hour"]
                # Calculate valid date for this forecast hour
                # For obs, use start of valid period: model_datetime + forecast_hour - 24
                obs_dt = model_datetime + timedelta(hours=int(fcst_hour)-24)
                obs_dt_str = obs_dt.strftime('%Y%m%d')
                if sid not in output["stations"]:
                    output["stations"][sid] = {}
                if fcst_hour not in output["stations"][sid]:
                    output["stations"][sid][fcst_hour] = {}
                for model, cols in model_columns.items():
                    output["stations"][sid][fcst_hour][model] = [row[col] for col in cols]
                # Add observation value for this valid date and sid
                output["stations"][sid][fcst_hour]["obs"] = obs_dicts.get(obs_dt_str, {}).get(sid, None)

        with open(json_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"JSON file saved as {json_file}")

        # Rsync the JSON file to the remote server with permission 755
        rsync_cmd = f"rsync -avz --chmod=F755 {json_file} chad.kahler@rsync3:/export/vhosts/dev/html/wrh/gefs/data/"
        print(f"Running: {rsync_cmd}")
        os.system(rsync_cmd)

    except Exception as e:
        print(f"Error converting to JSON: {e}")
    
    output = {
        "percentiles": percentiles,
        "stations": {}
    }
    
    try:
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sid"]
                fcst_hour = row["fcst_hour"]
                if sid not in output["stations"]:
                    output["stations"][sid] = {}
                if fcst_hour not in output["stations"][sid]:
                    output["stations"][sid][fcst_hour] = {}
                for model, cols in model_columns.items():
                    output["stations"][sid][fcst_hour][model] = [row[col] for col in cols]
                # Add observation value if available
                output["stations"][sid][fcst_hour]["obs"] = obs_dict.get(sid, None)
        
        with open(json_file, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"JSON file saved as {json_file}")
        
    except Exception as e:
        print(f"Error converting to JSON: {e}")


if __name__ == "__main__":
    """Entry point when script is run directly."""
    try:
        main()
        print("\nScript completed successfully!")
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nScript failed with error: {e}")
        sys.exit(1)
