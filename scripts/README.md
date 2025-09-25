# GEFS ML Training - Multiple Stations Runner

This directory contains scripts to run the GEFS ML training pipeline for multiple stations and forecast hours efficiently.

## Prerequisites

- **Conda Environment**: All scripts use the `gefsai` conda environment
- Make sure the `gefsai` environment is created and has all required packages installed
- The scripts will automatically activate the environment using `conda run -n gefsai`

## Files

- `single_gefs_ml_trainer.py` - Main Python training script (now supports command-line arguments)
- `run_multiple_stations.sh` - Bash script for running multiple stations (Linux/macOS)
- `run_multiple_stations.bat` - Batch script for running multiple stations (Windows CMD)
- `run_multiple_stations.ps1` - PowerShell script for running multiple stations (Windows PowerShell)
- `test_single_run.py` - Python helper script for testing individual runs

## Quick Start

### **Method 1: Simple Single Run (Recommended for Testing)**
```powershell
# Test a single station with the simple runner (avoids conda issues)
.\scripts\run_simple.ps1 -Station "KSEA" -ForecastHour "f024" -TargetVariable "tmax" -QuickMode
```

### **Method 2: Batch File (Windows - Most Reliable)**
```cmd
# Single station run with batch file
.\scripts\run_single_station.bat KSEA f024 tmax

# Or with custom models
.\scripts\run_single_station.bat KSEA f024 tmax gradient_boosting,lightgbm
```

### **Method 3: Multiple Stations (PowerShell)**
```powershell
# Full batch processing (may have conda encoding issues on some systems)
.\scripts\run_multiple_stations.ps1 -TargetVariable "tmax" -QuickMode $true
```

### **Method 4: Linux/macOS Users**
```bash
# Make script executable
chmod +x run_multiple_stations.sh

# Run batch processing
./run_multiple_stations.sh
```

## Configuration

Edit the configuration section in the appropriate runner script, or use PowerShell parameters.

### PowerShell Parameters (run_multiple_stations.ps1)

```powershell
# Basic usage with defaults
.\run_multiple_stations.ps1

# Custom target variable
.\run_multiple_stations.ps1 -TargetVariable "tmin"

# Enable GPU acceleration
.\run_multiple_stations.ps1 -UseGPU $true

# Disable quick mode (more thorough hyperparameter tuning)
.\run_multiple_stations.ps1 -QuickMode $false

# Custom models
.\run_multiple_stations.ps1 -Models "random_forest,xgboost"

# Combined parameters
.\run_multiple_stations.ps1 -TargetVariable "tmin" -QuickMode $false -Models "gradient_boosting,lightgbm"
```

### Key Parameters

- **TARGET_VARIABLE**: `"tmax"` or `"tmin"` (temperature maximum or minimum)
- **USE_GPU**: `true` or `false` (enable GPU acceleration if available)
- **QUICK_MODE**: `true` or `false` (faster training with smaller hyperparameter grids)
- **MODELS**: Comma-separated list of models to train

### Available Models

- `ols` - Ordinary Least Squares (Linear Regression)
- `ridge` - Ridge regression (L2 regularization)
- `lasso` - Lasso regression (L1 regularization)
- `stepwise` - Stepwise regression (feature selection + OLS)
- `random_forest` - Random Forest (always available)
- `gradient_boosting` - Scikit-learn Gradient Boosting (always available)
- `xgboost` - XGBoost (if available)
- `catboost` - CatBoost (if available)
- `lightgbm` - LightGBM (if available)
- `extra_trees` - Extra Trees (always available)

### Station List

Default stations included:
```
KSEA, KORD, KLAX, KBOI, KLAS, KDEN, KPHX, KSLC, KPDX, KGEG
```

### Forecast Hours

Default forecast hours:
```
f024, f048, f072
```

## Testing Individual Runs

Before running the full batch, test individual configurations:

```bash
# Basic test
python test_single_run.py KSEA f024 tmax

# Test with quick mode
python test_single_run.py KSEA f024 tmax --quick_mode

# Test with specific models
python test_single_run.py KSEA f024 tmax --quick_mode --models gradient_boosting,lightgbm
```

## Manual Single Runs

You can also run the main script directly (the scripts will automatically use the gefsai conda environment):

```bash
# Single station, single forecast hour
conda run -n gefsai python single_gefs_ml_trainer.py --station_id KSEA --forecast_hours f024 --target_variable tmax

# Multiple forecast hours
conda run -n gefsai python single_gefs_ml_trainer.py --station_id KSEA --forecast_hours f024,f048,f072 --target_variable tmax

# With quick mode and specific models
conda run -n gefsai python single_gefs_ml_trainer.py --station_id KSEA --forecast_hours f024 --target_variable tmax --quick_mode --model_names gradient_boosting,lightgbm,xgboost
```

## Command Line Arguments

The main Python script supports these arguments:

- `--station_id STATION` - Station ID to process (e.g., KSEA, KORD)
- `--forecast_hours HOURS` - Forecast hours, comma-separated (e.g., f024,f048)
- `--target_variable TARGET` - Target variable: 'tmax' or 'tmin'
- `--quick_mode` - Use quick mode with smaller hyperparameter grids
- `--model_names MODELS` - Models to train, comma-separated
- `--use_gpu` - Enable GPU acceleration if available

## Output

Each run produces:

1. **Model files**: Saved in `../models/` directory
   - `.joblib` files containing trained models
   - `.json` files containing metadata and configuration

2. **Plots**: Saved in `../plots/` directory
   - Scatter plots comparing predictions vs observations
   - Feature importance plots
   - Residual analysis plots

3. **Logs**: For batch runs, logs are saved in `../logs/` directory

## File Naming Convention

Files are named using this pattern:
```
gefs_ml_{target}_{station}_{fhr}_{model}_{timestamp}
```

For example:
```
gefs_ml_tmax_KSEA_fhr24_lightgbm_20250925_143022.joblib
gefs_ml_tmax_KSEA_fhr24_lightgbm_20250925_143022.json
```

Latest models are also saved with `_latest` suffix for easy access.

## Performance Tips

1. **Quick Mode**: Use `--quick_mode` for faster training during development/testing
2. **GPU Acceleration**: Enable `--use_gpu` if you have compatible hardware
3. **Model Selection**: Start with `gradient_boosting,lightgbm` for good performance/speed balance
4. **Batch Processing**: The runner scripts automatically handle resource management and logging

## Troubleshooting

### **Common Issues**

1. **Unicode Encoding Errors (Windows)**
   ```
   UnicodeEncodeError: 'charmap' codec can't encode character
   ```
   **Solution**: Use the simple runners instead of the full batch script:
   ```powershell
   .\scripts\run_simple.ps1 -Station "KSEA" -ForecastHour "f024" -TargetVariable "tmax" -QuickMode
   ```

2. **Conda Environment Not Found**
   ```
   ERROR: gefsai conda environment not found
   ```
   **Solution**: Create the environment from the YAML file:
   ```powershell
   conda env create -f gefsai.yml
   ```

3. **Import Errors**: Optional packages (cuML, XGBoost, etc.) will show warnings but won't stop execution

4. **Memory Issues**: Reduce the number of parallel jobs in the Python script if needed

5. **Missing Data**: Check that data files exist in the expected paths (`N:/data/gefs-ml/`)

### **Alternative Execution Methods**

If you're having issues with the conda environment or PowerShell scripts:

1. **Manual conda activation**:
   ```powershell
   conda activate gefsai
   python .\scripts\single_gefs_ml_trainer.py --station_id KSEA --forecast_hours f024 --target_variable tmax --quick_mode
   ```

2. **Using regular Python** (if gefsai environment has issues):
   ```powershell
   python .\scripts\single_gefs_ml_trainer.py --station_id KSEA --forecast_hours f024 --target_variable tmax --quick_mode
   ```

3. **Batch file approach** (Windows):
   ```cmd
   .\scripts\run_single_station.bat KSEA f024 tmax
   ```

## Example Workflow

1. Test a single configuration:
   ```bash
   python test_single_run.py KSEA f024 tmax --quick_mode
   ```

2. Edit the runner script to configure your desired stations and parameters

3. Run the full batch:
   ```bash
   # Windows
   run_multiple_stations.bat
   
   # Linux/macOS
   ./run_multiple_stations.sh
   ```

4. Monitor progress in the console and check the log file for detailed results

## Summary

This multi-station runner system provides:

1. **Automated batch processing** - Run training for multiple stations and forecast hours automatically
2. **Cross-platform support** - Bash, Windows Batch, and PowerShell versions
3. **Comprehensive logging** - All output captured to timestamped log files
4. **Error handling** - Failed runs don't stop the batch, summary shows success/failure counts
5. **Easy testing** - Test individual configurations before running full batches
6. **Flexible configuration** - Command-line arguments override default settings
7. **Resource management** - Small delays between runs prevent resource conflicts

The system keeps individual Python runs short and refreshes the environment for each station, ensuring optimal performance and reliability.