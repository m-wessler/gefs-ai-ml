#!/home/chad.kahler/anaconda3/envs/dev/bin/python

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor  # Commented out as no longer needed
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import joblib
from datetime import datetime, timedelta
from xgboost import XGBRegressor


# Configuration
stations_file = '/home/chad.kahler/gefs/wr_obs_filtered_index.csv'
start_date = "2000-01-01"  # Start date for the analysis period
end_date = "2006-12-31"  # End date for the analysis period
target_variable = "tmax_obs"  # Target variable
forecast_hours = [6, 12, 24, 48, 72, 120, 168, 240]
# forecast_hours = [6]

variables = [
    "pres_msl", "pres_sfc",
    "hgt_pres_925", "hgt_pres_850", "hgt_pres_700",
    "tmp_2m", "tmp_pres_925", "tmp_pres_850", "tmp_pres_700", "tmin_2m", "tmax_2m",
    "spfh_2m", "spfh_pres_925", "spfh_pres_850", "spfh_pres_700",
    "gust_sfc", "ugrd_hgt", "ugrd_pres_925", "ugrd_pres_850", "ugrd_pres_700",
    "vgrd_hgt", "vgrd_pres_925", "vgrd_pres_850", "vgrd_pres_700",
    "dswrf_sfc", "dlwrf_sfc", "uswrf_sfc", "ulwrf_sfc", "lhtfl_sfc", "shtfl_sfc", "gflux_sfc",
    "cape_sfc", "cin_sfc", "hgt_ceiling",
    "soilw_bgrnd",
    "tcdc_eatm",
    "uflx_sfc", "vflx_sfc"
]
data_dir = "/nas/stid/data/gefs_reforecast/station_data/combined"  # Directory containing the input file
base_output_dir = "./pca_results"  # Base directory for output
target_correlation_threshold = 1.1  # Threshold for excluding variables correlated with target
loading_threshold = 0.1  # Threshold for including variables in PCA loading plots
variance_threshold = 0.8  # Cumulative variance threshold for plotting PC loading bar plots (80%)
similar_corr_threshold = 0.05  # Threshold for considering correlations with target "similar"

exclude_vars_dict = {
    "tmax_obs": ["tmin_2m", "tmp_2m"],
}


# Function to create error histogram (ME, MAE, or RMSE)
def plot_error_histogram(actual, predicted, output_dir, base_filename, station_id, fhour, title_prefix, error_type='mae'):
    """
    Create a histogram of errors (ME, MAE, or RMSE) and annotate with the mean error, mean absolute error, or root mean squared error.
    Save with '_me.png', '_mae.png', or '_rmse.png' appended to the base filename.
    error_type: 'me' for Mean Error, 'mae' for Mean Absolute Error, 'rmse' for Root Mean Squared Error
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
        errors = (predicted - actual) ** 2  # Squared errors for histogram
        mean_error = root_mean_squared_error(actual, predicted)  # RMSE
        label = f'RMSE = {mean_error:.2f} °C'
        filename_suffix = '_rmse.png'
        xlabel = 'Squared Error (°C²)'
        title_suffix = 'RMSE'
    
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=30, kde=True, color='purple')
    plt.axvline(mean_error if error_type != 'rmse' else mean_error**2, color='red', linestyle='--', label=label)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(f'{title_prefix} {title_suffix} Distribution\n(Station: {station_id}, Forecast Hour: {fhour})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"{base_filename}{filename_suffix}")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"{title_suffix} histogram saved to {plot_file}")

# Function to extract pressure level from variable names
def get_pressure_level(var_name):
    """Extract the pressure level (in hPa) from a variable name, if applicable."""
    try:
        if '_pres_' in var_name:
            return int(var_name.split('_pres_')[-1])
        return None
    except:
        return None

# Function to extract base parameter type from variable names
def get_base_parameter(var_name):
    """Extract the base parameter type from a variable name."""
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

# Function to extract and preprocess data from the single file
def extract_station_data(file_path, station_id, start_date, end_date, variables, target_variable):
    try:
        df = pd.read_csv(file_path)
        df["valid_datetime"] = pd.to_datetime(df["valid_datetime"])
        df = df.drop_duplicates(subset=["valid_datetime", "perturbation"], keep="first")
        df = df[df["sid"] == station_id]
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        df = df[(df["valid_datetime"] >= start) & (df["valid_datetime"] <= end + timedelta(days=1))]
        if df.empty:
            print(f"No data found for station {station_id} in {file_path} for the specified date range.")
            return None
        group_cols = ["sid", "valid_datetime"]
        agg_cols = variables + [target_variable]
        df = df[group_cols + agg_cols].groupby(["sid", "valid_datetime"]).mean().reset_index()
        cols = ["sid", "valid_datetime"] + variables + [target_variable]
        if not all(col in df.columns for col in cols):
            print(f"Missing columns in {file_path}: {[col for col in cols if col not in df.columns]}")
            return None
        data = df[cols].copy()
        data["date"] = data["valid_datetime"]
        return data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load station locations
stations_df = pd.read_csv(stations_file)

for idx, row in stations_df.iterrows():
    station_id = row['sid']
    if station_id in ["KSLC", "KPHX", "KLAX", "KLAS", "KSEA"]:
    # if station_id in ["KPHX"]:
        for fhour in forecast_hours:
            input_file = f"{station_id}_{start_date[:4]}_{end_date[:4]}_f{fhour:03d}.csv"
            print(f"\nProcessing station: {station_id} for forecast hour: {fhour}")
            output_dir = os.path.join(base_output_dir, station_id, f"f{fhour:03d}")
            os.makedirs(output_dir, exist_ok=True)

            file_path = os.path.join(data_dir, input_file)
            if not os.path.exists(file_path):
                print(f"Input file {file_path} not found.")
                exit(1)

            print(f"Loading data from: {file_path}")
            df = extract_station_data(file_path, station_id, start_date, end_date, variables, target_variable)

            if df is None or df.empty:
                print(f"No data collected for station {station_id} at forecast hour {fhour}. Exiting.")
                exit(1)

            df["forecast_hour"] = fhour

            # --- Scatter plot of tmax_2m vs tmax_obs ---
            if 'tmax_2m' in df.columns and 'tmax_obs' in df.columns:
                plot_df = df[['tmax_2m', 'tmax_obs']].dropna()
                if not plot_df.empty:
                    r2 = r2_score(plot_df['tmax_obs'], plot_df['tmax_2m'])
                    plt.figure(figsize=(8, 6))
                    plt.scatter(plot_df['tmax_obs'], plot_df['tmax_2m'], alpha=0.5, color='blue', label='GEFS tmax_2m')
                    plot_max = max(plot_df['tmax_2m'].max(), plot_df['tmax_obs'].max())
                    plot_min = min(plot_df['tmax_2m'].min(), plot_df['tmax_obs'].min())
                    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2)
                    plt.xlabel("Observed tmax_obs (C)")
                    plt.ylabel("GEFS tmax_2m (C)")
                    plt.title(f"GEFS tmax_2m vs Observed tmax_obs\n(Station: {station_id}, Forecast Hour: {fhour}, R² = {r2:.4f})")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plot_file = os.path.join(output_dir, f"tmax_2m_vs_tmax_obs_{station_id}_f{fhour:03d}.png")
                    plt.savefig(plot_file, dpi=300)
                    plt.close()
                    print(f"Scatter plot of tmax_2m vs tmax_obs saved to {plot_file}")

                    # --- ME histogram for tmax_2m vs tmax_obs ---
                    plot_error_histogram(
                        actual=plot_df['tmax_obs'],
                        predicted=plot_df['tmax_2m'],
                        output_dir=output_dir,
                        base_filename=f"tmax_2m_vs_tmax_obs_{station_id}_f{fhour:03d}",
                        station_id=station_id,
                        fhour=fhour,
                        title_prefix="GEFS tmax_2m vs Observed tmax_obs",
                        error_type='me'
                    )

                    # --- RMSE histogram for tmax_2m vs tmax_obs ---
                    plot_error_histogram(
                        actual=plot_df['tmax_obs'],
                        predicted=plot_df['tmax_2m'],
                        output_dir=output_dir,
                        base_filename=f"tmax_2m_vs_tmax_obs_{station_id}_f{fhour:03d}",
                        station_id=station_id,
                        fhour=fhour,
                        title_prefix="GEFS tmax_2m vs Observed tmax_obs",
                        error_type='rmse'
                    )
                else:
                    print(f"No valid data for tmax_2m vs tmax_obs scatter plot for {station_id} at forecast hour {fhour}")
            else:
                print(f"Columns tmax_2m and/or tmax_obs missing for {station_id} at forecast hour {fhour}")

            with open(os.path.join(output_dir, f"station_id_{station_id}_f{fhour:03d}.txt"), "w") as f:
                f.write(f"Station ID: {station_id}\nForecast Hour: {fhour}")
            print(f"Station ID and forecast hour info saved to station_id_{station_id}_f{fhour:03d}.txt")

            df = df.dropna()
            correlation_matrix = df[variables + [target_variable]].corr(method="pearson")
            tmax_correlations = correlation_matrix[target_variable].drop(target_variable)

            # Step 1: Exclude manually specified variables
            exclude_vars = exclude_vars_dict.get(target_variable, [])
            if exclude_vars:
                print(f"\nExcluding manually specified variables for target {target_variable} for {station_id}: {exclude_vars}")
            else:
                print(f"\nNo manually specified variables to exclude for target {target_variable} for {station_id}")

            feature_cols = [var for var in variables if var not in exclude_vars]
            print(f"\nFeatures after excluding manually specified variables: {feature_cols}")

            # Step 2: Select one variation per parameter type based on highest correlation with target
            param_groups = {}
            for var in feature_cols:
                base_param = get_base_parameter(var)
                if base_param not in param_groups:
                    param_groups[base_param] = []
                param_groups[base_param].append(var)

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
                    candidates = [var for var, corr in correlations if max_corr - corr <= similar_corr_threshold]
                    
                    if len(candidates) == 1:
                        selected_var = candidates[0]
                        selected_vars.append(selected_var)
                        print(f"Selected {selected_var} for {base_param} with highest target correlation ({tmax_correlations[selected_var]:.4f}) among {valid_vars}")
                    else:
                        candidate_levels = [(var, get_pressure_level(var)) for var in candidates]
                        candidate_levels = [(var, level if level is not None else 10000) for var, level in candidate_levels]
                        candidate_levels.sort(key=lambda x: x[1])
                        selected_var = candidate_levels[0][0]
                        selected_vars.append(selected_var)
                        print(f"Selected {selected_var} for {base_param} with correlation {tmax_correlations[selected_var]:.4f} (level: {get_pressure_level(selected_var)}) among {valid_vars} (similar correlations)")
                    
                    excluded_vars.update(set(var_list) - {selected_var})

            feature_cols = selected_vars
            print(f"\nFeatures after selecting one variation per parameter type: {feature_cols}")

            # Step 3: Exclude variables highly correlated with the target
            high_corr_target_vars = tmax_correlations[feature_cols][abs(tmax_correlations[feature_cols]) >= target_correlation_threshold].index.tolist()
            if high_corr_target_vars:
                print(f"\nExcluding variables with |correlation| >= {target_correlation_threshold} with {target_variable} for {station_id}: {high_corr_target_vars}")
                excluded_vars.update(high_corr_target_vars)
            else:
                print(f"\nNo variables with |correlation| >= {target_correlation_threshold} with {target_variable} for {station_id}")

            feature_cols = [var for var in feature_cols if var not in high_corr_target_vars]
            print(f"\nFinal features retained for PCA: {feature_cols}")

            # Save all excluded variables
            all_excluded_vars = list(excluded_vars)
            with open(os.path.join(output_dir, f"excluded_variables_{station_id}_f{fhour:03d}.txt"), "w") as f:
                f.write("\n".join(all_excluded_vars) if all_excluded_vars else "None")
            print(f"Excluded variables saved to excluded_variables_{station_id}_f{fhour:03d}.txt")

            print(f"\nFull Correlation Matrix for {station_id} (rounded to 2 decimals):")
            print(correlation_matrix.round(2))

            correlation_matrix.to_csv(os.path.join(output_dir, f"correlation_matrix_{station_id}_f{fhour:03d}.csv"))
            print(f"Correlation matrix saved to correlation_matrix_{station_id}_f{fhour:03d}.csv")

            correlation_pairs = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            correlation_pairs = correlation_pairs.stack().reset_index()
            correlation_pairs.columns = ["Variable_1", "Variable_2", "Correlation"]
            correlation_pairs = correlation_pairs.dropna()
            correlation_pairs = correlation_pairs.sort_values(by="Correlation", key=abs, ascending=False)
            correlation_pairs.to_csv(os.path.join(output_dir, f"correlation_pairs_{station_id}_f{fhour:03d}.csv"), index=False)
            print(f"Pairwise correlations saved to correlation_pairs_{station_id}_f{fhour:03d}.csv")

            tmax_correlations.to_csv(os.path.join(output_dir, f"tmax_correlations_{station_id}_f{fhour:03d}.csv"))
            print(f"Correlations with tmax_obs saved to tmax_correlations_{station_id}_f{fhour:03d}.csv")

            plt.figure(figsize=(20, 16))
            sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f")
            plt.title(f"Correlation Matrix of All Variables (Station: {station_id}, Forecast Hour: {fhour})")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"correlation_matrix_heatmap_{station_id}_f{fhour:03d}.png"), dpi=300)
            plt.close()
            print(f"Correlation matrix heatmap saved to correlation_matrix_heatmap_{station_id}_f{fhour:03d}.png")

            top_n = 10
            print(f"\nTop {top_n} pairwise correlations for {station_id} (absolute value):")
            print(correlation_pairs.head(top_n))

            print(f"\nTop {top_n} variables most correlated with tmax_obs for {station_id} (absolute value):")
            top_tmax_correlations = tmax_correlations.abs().sort_values(ascending=False).head(top_n)
            print(top_tmax_correlations)

            if not feature_cols:
                print(f"No features remaining for PCA after filtering for {station_id}. Exiting.")
                exit(1)

            X = df[feature_cols].values
            y = df[target_variable].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            n_components = min(len(feature_cols), X_scaled.shape[0])
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

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

            pca_results.to_csv(os.path.join(output_dir, f"pca_variance_{station_id}_f{fhour:03d}.csv"), index=False)
            loadings.to_csv(os.path.join(output_dir, f"pca_loadings_{station_id}_f{fhour:03d}.csv"))
            df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
            df_pca["tmax_obs"] = y
            df_pca["sid"] = df["sid"]
            df_pca["date"] = df["date"]
            df_pca["forecast_hour"] = df["forecast_hour"]
            df_pca.to_csv(os.path.join(output_dir, f"pca_transformed_data_{station_id}_f{fhour:03d}.csv"), index=False)

            num_pcs_to_plot = min(15, n_components)
            plt.figure(figsize=(12, 6))
            plt.bar(range(1, num_pcs_to_plot + 1), explained_variance_ratio[:num_pcs_to_plot], alpha=0.5, align='center', label='Individual')
            plt.plot(range(1, num_pcs_to_plot + 1), cumulative_variance[:num_pcs_to_plot], marker='o', color='r', label='Cumulative')
            plt.xlabel('Principal Component')
            plt.ylabel('Variance Explained')
            plt.title(f'Variance Explained by First {num_pcs_to_plot} PCs for {station_id}')
            plt.xticks(range(1, num_pcs_to_plot + 1))
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'variance_explained_{station_id}_f{fhour:03d}.png'), dpi=300)
            plt.close()
            print(f"Variance explained plot saved to variance_explained_{station_id}_f{fhour:03d}.png")

            correlations = df_pca.corr(numeric_only=True)["tmax_obs"].drop("tmax_obs")
            print(f"\nCorrelation of PCs with tmax_obs for {station_id}:")
            print(correlations)

            best_pc = correlations.abs().idxmax()
            best_corr = correlations[best_pc]
            print(f"\nBest PC for tmax_obs correlation: {best_pc} (Correlation: {best_corr:.4f})")

            pcs_to_plot = np.where(cumulative_variance >= variance_threshold)[0]
            if len(pcs_to_plot) > 0:
                pcs_to_plot = pcs_to_plot[0] + 1
            else:
                pcs_to_plot = n_components
            print(f"Generating bar plots for {pcs_to_plot} PCs to explain at least {variance_threshold*100}% variance (Cumulative Variance: {cumulative_variance[pcs_to_plot-1]:.4f})")

            for i in range(pcs_to_plot):
                pc = f"PC{i+1}"
                plt.figure(figsize=(14, 6))
                pc_loadings = loadings[pc]
                pc_loadings = pc_loadings[abs(pc_loadings) > loading_threshold]
                if not pc_loadings.empty:
                    pc_loadings = pc_loadings.sort_values(key=abs, ascending=False)
                    pc_loadings.plot(kind="bar")
                    plt.title(f"PCA Loadings for {pc} (|Loading| > {loading_threshold}) (Station: {station_id}, Forecast Hour: {fhour})")
                    plt.xlabel("Variables")
                    plt.ylabel("Loading Value")
                    plt.xticks(rotation=45, ha="right")
                    plt.grid(axis="y", linestyle="--", alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"pca_loadings_bar_{pc}_{station_id}_f{fhour:03d}.png"), dpi=300)
                    plt.close()
                    print(f"Bar plot for {pc} loadings saved to pca_loadings_bar_{pc}_{station_id}_f{fhour:03d}.png")
                    top_n_loadings = 10
                    top_loadings = pc_loadings.abs().sort_values(ascending=False).head(top_n_loadings)
                    print(f"\nTop {top_n_loadings} variables contributing to {pc} for {station_id} (|loading| > {loading_threshold}):")
                    print(top_loadings)
                else:
                    print(f"No variables with |loading| > {loading_threshold} for {pc} in {station_id}. Skipping bar plot.")

            print(f"\nPCA Results for {station_id} (using {len(feature_cols)} features):")
            print(pca_results)
            print(f"\nComponent Loadings for {station_id}:")
            print(loadings)

            X_train, X_test, y_train, y_test = train_test_split(
                X_pca, y, test_size=0.2, random_state=42
            )
            print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

            # Linear Regression
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            print(f"\nLinear Regression for {station_id}, Forecast Hour {fhour}:")
            print(f"Training R² Score: {r2_train:.4f}")
            print(f"Testing R² Score: {r2_test:.4f}")
            print(f"Regression coefficients for PCs: {model.coef_}")

            with open(os.path.join(output_dir, f"regression_results_linear_{station_id}_f{fhour:03d}.txt"), "w") as f:
                f.write(f"Training R² Score for predicting tmax_obs: {r2_train:.4f}\n")
                f.write(f"Testing R² Score for predicting tmax_obs: {r2_test:.4f}\n")
                f.write("Regression coefficients for PCs:\n")
                for i, coef in enumerate(model.coef_, 1):
                    f.write(f"PC{i}: {coef:.4f}\n")
            print(f"Regression results saved to regression_results_linear_{station_id}_f{fhour:03d}.txt")

            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred_test, alpha=0.5)
            plot_max = max(y_test.max(), y_pred_test.max())
            plot_min = min(y_test.min(), y_pred_test.min())
            plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2)
            plt.xlabel("Actual tmax_obs")
            plt.ylabel("Predicted tmax_obs")
            plt.title(f"Actual vs. Predicted tmax_obs (Test Set, Station: {station_id}, Forecast Hour: {fhour}, R² = {r2_test:.4f})")
            plt.grid(True)
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"regression_scatter_linear_{station_id}_f{fhour:03d}.png")
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Scatter plot saved to {plot_file}")

            # --- MAE histogram for Linear Regression ---
            plot_error_histogram(
                actual=y_test,
                predicted=y_pred_test,
                output_dir=output_dir,
                base_filename=f"regression_scatter_linear_{station_id}_f{fhour:03d}",
                station_id=station_id,
                fhour=fhour,
                title_prefix="Linear Regression Actual vs Predicted tmax_obs",
                error_type='mae'
            )

            # --- RMSE histogram for Linear Regression ---
            plot_error_histogram(
                actual=y_test,
                predicted=y_pred_test,
                output_dir=output_dir,
                base_filename=f"regression_scatter_linear_{station_id}_f{fhour:03d}",
                station_id=station_id,
                fhour=fhour,
                title_prefix="Linear Regression Actual vs Predicted tmax_obs",
                error_type='rmse'
            )

            # # Random Forest (Commented Out)
            # param_grid = {
            #     'n_estimators': [50, 100, 200],
            #     'max_depth': [None, 10, 20]
            # }
            # grid_search = GridSearchCV(
            #     RandomForestRegressor(random_state=42),
            #     param_grid,
            #     cv=5,
            #     scoring='r2',
            #     n_jobs=-1
            # )
            # grid_search.fit(X_train, y_train)
            # model = grid_search.best_estimator_
            # print(f"\nBest parameters for {station_id} at forecast hour {fhour}: {grid_search.best_params_}")
            # print(f"Best cross-validated R² score (training): {grid_search.best_score_:.4f}")
            #
            # y_pred_train = model.predict(X_train)
            # y_pred_test = model.predict(X_test)
            # r2_train = r2_score(y_train, y_pred_train)
            # r2_test = r2_score(y_test, y_pred_test)
            # print(f"\nRandom Forest for {station_id}, Forecast Hour {fhour}:")
            # print(f"Training R² Score: {r2_train:.4f}")
            # print(f"Testing R² Score: {r2_test:.4f}")
            #
            # model_file = os.path.join(output_dir, f"rf_model_{station_id}_f{fhour:03d}.pkl")
            # joblib.dump(model, model_file)
            # print(f"Saved Random Forest model to {model_file}")
            #
            # with open(os.path.join(output_dir, f"regression_results_rf_{station_id}_f{fhour:03d}.txt"), "w") as f:
            #     f.write(f"Training R² Score for predicting {target_variable}: {r2_train:.4f}\n")
            #     f.write(f"Testing R² Score for predicting {target_variable}: {r2_test:.4f}\n")
            #     f.write(f"Best parameters: {grid_search.best_params_}\n")
            #     f.write(f"Best cross-validated R² score (training): {grid_search.best_score_:.4f}\n")
            #     f.write("Feature importances for PCs:\n")
            #     for i, importance in enumerate(model.feature_importances_, 1):
            #         f.write(f"PC{i}: {importance:.4f}\n")
            # print(f"Regression results saved to regression_results_rf_{station_id}_f{fhour:03d}.txt")
            #
            # plt.figure(figsize=(8, 6))
            # plt.scatter(y_test, y_pred_test, alpha=0.5)
            # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            # plt.xlabel(f"Actual {target_variable}")
            # plt.ylabel(f"Predicted {target_variable}")
            # plt.title(f"Actual vs. Predicted {target_variable} (Test Set, Station: {station_id}, Forecast Hour: {fhour:03d}, R² = {r2_test:.4f})")
            # plt.grid(True)
            # plt.tight_layout()
            # plot_file = os.path.join(output_dir, f"regression_scatter_rf_{station_id}_f{fhour:03d}.png")
            # plt.savefig(plot_file, dpi=300)
            # plt.close()
            # print(f"Scatter plot saved to {plot_file}")
            #
            # # --- MAE histogram for Random Forest ---
            # plot_error_histogram(
            #     actual=y_test,
            #     predicted=y_pred_test,
            #     output_dir=output_dir,
            #     base_filename=f"regression_scatter_rf_{station_id}_f{fhour:03d}",
            #     station_id=station_id,
            #     fhour=fhour,
            #     title_prefix="Random Forest Actual vs Predicted tmax_obs",
            #     error_type='mae'
            # )
            #
            # # --- RMSE histogram for Random Forest ---
            # plot_error_histogram(
            #     actual=y_test,
            #     predicted=y_pred_test,
            #     output_dir=output_dir,
            #     base_filename=f"regression_scatter_rf_{station_id}_f{fhour:03d}",
            #     station_id=station_id,
            #     fhour=fhour,
            #     title_prefix="Random Forest Actual vs Predicted tmax_obs",
            #     error_type='rmse'
            # )

            ##### XGBoost doesn't work well wtih GridSearchCSV
            # # XGBoost with XGBRegressorWrapper
            # param_grid = {
            #     'n_estimators': [50, 100, 200],
            #     'max_depth': [3, 6, 10],
            #     'learning_rate': [0.01, 0.1, 0.3]
            # }
            # grid_search = GridSearchCV(
            #     xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
            #     param_grid,
            #     cv=5,
            #     scoring='r2',
            #     n_jobs=-1
            # )
            # grid_search.fit(X_train, y_train)    
            # model = grid_search.best_estimator_
            # print(f"\nBest parameters for {station_id} at forecast hour {fhour}: {grid_search.best_params_}")
            # print(f"Best cross-validated R² score (training): {grid_search.best_score_:.4f}")

            # Create an XGBRegressor
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            
            # Fit the model on the training data
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            print(f"\nXGBoost for {station_id}, Forecast Hour {fhour}:")
            print(f"Training R² Score: {r2_train:.4f}")
            print(f"Testing R² Score: {r2_test:.4f}")

            model_file = os.path.join(output_dir, f"xgb_model_{station_id}_f{fhour:03d}.pkl")
            joblib.dump(model, model_file)
            print(f"Saved XGBoost model to {model_file}")

            with open(os.path.join(output_dir, f"regression_results_xgb_{station_id}_f{fhour:03d}.txt"), "w") as f:
                f.write(f"Training R² Score for predicting {target_variable}: {r2_train:.4f}\n")
                f.write(f"Testing R² Score for predicting {target_variable}: {r2_test:.4f}\n")
                # f.write(f"Best parameters: {grid_search.best_params_}\n")     ##### NOT USING GridSearchCSV
                # f.write(f"Best cross-validated R² score (training): {grid_search.best_score_:.4f}\n")
                f.write("Feature importances for PCs:\n")
                for i, importance in enumerate(model.feature_importances_, 1):  # Access underlying XGBRegressor
                    f.write(f"PC{i}: {importance:.4f}\n")
            print(f"Regression results saved to regression_results_xgb_{station_id}_f{fhour:03d}.txt")

            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred_test, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel(f"Actual {target_variable}")
            plt.ylabel(f"Predicted {target_variable}")
            plt.title(f"Actual vs. Predicted {target_variable} (Test Set, Station: {station_id}, Forecast Hour: {fhour:03d}, R² = {r2_test:.4f})")
            plt.grid(True)
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"regression_scatter_xgb_{station_id}_f{fhour:03d}.png")
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Scatter plot saved to {plot_file}")

            # --- MAE histogram for XGBoost ---
            plot_error_histogram(
                actual=y_test,
                predicted=y_pred_test,
                output_dir=output_dir,
                base_filename=f"regression_scatter_xgb_{station_id}_f{fhour:03d}",
                station_id=station_id,
                fhour=fhour,
                title_prefix="XGBoost Actual vs Predicted tmax_obs",
                error_type='mae'
            )

            # --- RMSE histogram for XGBoost ---
            plot_error_histogram(
                actual=y_test,
                predicted=y_pred_test,
                output_dir=output_dir,
                base_filename=f"regression_scatter_xgb_{station_id}_f{fhour:03d}",
                station_id=station_id,
                fhour=fhour,
                title_prefix="XGBoost Actual vs Predicted tmax_obs",
                error_type='rmse'
            )