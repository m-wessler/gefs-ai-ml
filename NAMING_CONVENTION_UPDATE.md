# File Naming Convention Update Summary

## Changes Made to `scripts/single_gefs_ml_trainer.py`

### 1. New Utility Functions Added

- `generate_station_identifier()`: Creates station ID string
  - Single station: Uses the station ID directly (e.g., "KSLC")
  - Multiple stations: Uses "multi_{count}" format (e.g., "multi_3", "multi_500")

- `generate_forecast_hour_identifier()`: Creates forecast hour string  
  - Single hour: "fhr{hour}" (e.g., "fhr24" for f024)
  - Multiple hours: "fhr{min}-{max}" (e.g., "fhr24-72" for f024,f048,f072)

- `generate_filename_base()`: Combines all elements into base filename
  - Format: `gefs_ml_{target}_{station_id}_{fhr_id}_{model_name}`

### 2. Updated Functions

**Model & Metadata Saving:**
- `save_model_and_metadata()`: Now uses new naming convention
- Model files: `{base}_{timestamp}.joblib`
- Metadata files: `{base}_{timestamp}.json` 
- Latest files: `{base}_latest.{ext}`

**Plotting Functions:**
- `create_scatter_plot()`: Added `best_model_name` parameter
- `create_feature_importance_plot()`: Added `best_model_name` parameter  
- `create_residuals_plot()`: Added `best_model_name` parameter
- All plot files now use: `{base}_{plot_type}_{timestamp}.png`

**Model Loading:**
- `load_model_and_metadata()`: Updated to handle new naming with glob patterns

### 3. Example Filenames

**Single Station (KSLC), Single Hour (f024), XGBoost:**
- Model: `gefs_ml_tmax_KSLC_fhr24_xgboost_20250924_120000.joblib`
- Metadata: `gefs_ml_tmax_KSLC_fhr24_xgboost_20250924_120000.json`
- Plots: `gefs_ml_tmax_KSLC_fhr24_xgboost_evaluation_20250924_120000.png`

**Multiple Stations (5), Multiple Hours (f024-f072), Random Forest:**
- Model: `gefs_ml_tmin_multi_5_fhr24-72_random_forest_20250924_120000.joblib`
- Metadata: `gefs_ml_tmin_multi_5_fhr24-72_random_forest_20250924_120000.json`  
- Plots: `gefs_ml_tmin_multi_5_fhr24-72_random_forest_feature_analysis_20250924_120000.png`

### 4. Benefits

- **Clear identification**: Can immediately see station(s), forecast hours, and model from filename
- **No ambiguity**: Different training runs have clearly distinct names
- **Sortable**: Files naturally sort by target → station → forecast hour → model
- **Backwards compatible**: Old loading function has fallback for new naming

### 5. Breaking Changes

- Old "_latest" files may not be found by default loading function
- Scripts expecting old naming convention will need updates
- Glob patterns may be needed to find specific files programmatically

The implementation maintains all existing functionality while providing much more informative and organized file naming.