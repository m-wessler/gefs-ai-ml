# GEFS ML Data Alignment Improvements

## Problem Statement
When aggregating across forecast hours (e.g., using both f024 and f048), the original code had data alignment issues that were evident in NBM verification scatterplots. The main problems were:

1. **Multiple forecast hours per verification time**: Multiple forecast records (f024, f048) could match the same URMA observation time, but were not properly aligned with corresponding NBM forecasts.

2. **Loose time window matching**: The 12-hour window matching allowed mismatched GEFS-NBM pairs where GEFS and NBM forecasts were from different forecast cycles or lead times.

3. **Lack of forecast lead time tracking**: No information about forecast lead time was preserved, making it difficult to diagnose alignment issues.

## Solutions Implemented

### 1. Improved Time Matching Logic (`create_time_matched_dataset`)

**Before**: 
- Combined forecast and NBM data by iterating through all forecast matches
- Could pair GEFS f024 with NBM f048 for the same verification time
- No validation that GEFS and NBM were from the same forecast cycle

**After**:
- **Exact forecast time alignment**: For each forecast valid time, find the corresponding NBM forecast for the exact same time
- **Skip mismatched pairs**: If no NBM forecast exists for a given GEFS forecast time, skip that GEFS forecast
- **Track forecast lead time**: Add `forecast_lead_hours` column to track how far ahead each forecast is
- **Better statistics**: Report match rates and forecast hour distribution

### 2. Enhanced Data Alignment Verification (`verify_data_alignment`)

**New Features**:
- **Lead time-specific correlation checking**: Check GEFS-NBM correlation separately for each forecast lead time
- **Forecast hour distribution reporting**: Show how many forecasts exist at each lead time
- **Enhanced sample display**: Show forecast lead times in sample alignment checks
- **Warning system**: Alert when correlations are too low, indicating alignment issues

### 3. Forecast Hour Aggregation Options (`aggregate_forecast_hours_for_evaluation`)

**Three strategies** to handle multiple forecast hours per verification time:

1. **'separate' (default)**: Keep each forecast hour as separate training examples
   - Pros: More training data, preserves forecast hour information
   - Cons: May have slight temporal correlation between samples

2. **'ensemble'**: Average multiple forecast hours for same verification time
   - Pros: Reduces potential overfitting from correlated samples
   - Cons: Less training data, loses forecast hour information

3. **'best_lead'**: Keep only forecast closest to 24-hour lead time
   - Pros: Uses optimal lead time, avoids correlation issues
   - Cons: Significantly less training data

### 4. Configuration Management

**New Configuration Variable**:
```python
FORECAST_HOUR_AGGREGATION = 'separate'  # Options: 'separate', 'ensemble', 'best_lead'
```

This setting is:
- Saved in model metadata for reproducibility
- Configurable without code changes
- Documented with pros/cons for each option

## Key Code Changes

### 1. Modified `create_time_matched_dataset()`:
```python
# OLD: Loose pairing that could mismatch GEFS and NBM
for (valid_dt, sid), forecast_row in forecast_matches.iterrows():
    try:
        nbm_row = nbm_matches.loc[(valid_dt, sid)]
        # ... combine data
    except KeyError:
        continue

# NEW: Exact time alignment ensuring GEFS-NBM match
forecast_times = forecast_matches.index.get_level_values('valid_datetime').unique()
for forecast_time in forecast_times:
    try:
        forecast_row = forecast_matches.loc[(forecast_time, station)]
        nbm_row = nbm_matches.loc[(forecast_time, station)]  # Exact match required
        # ... combine with lead time info
    except KeyError:
        continue  # Skip if no exact NBM match
```

### 2. Enhanced verification with lead time analysis:
```python
# Check correlation by forecast hour
for lead_hour in df['forecast_lead_hours'].unique():
    subset = df[df['forecast_lead_hours'] == lead_hour]
    correlation = subset[gefs_col].corr(subset[nbm_col])
    print(f"  {lead_hour:6.1f}h: GEFS-NBM correlation = {correlation:.3f}")
    if correlation < 0.8:
        print(f"    WARNING: Low correlation at {lead_hour}h suggests alignment issues!")
```

## Expected Improvements

1. **Better NBM verification plots**: GEFS and NBM should now be properly aligned, showing more realistic baseline performance
2. **Improved model performance**: Properly aligned training data should lead to better model learning
3. **Diagnostic capabilities**: Can now identify and fix alignment issues more easily
4. **Flexibility**: Can experiment with different aggregation strategies based on use case

## Monitoring Alignment Quality

The enhanced verification will now report:
- GEFS-NBM correlation by forecast hour (should be > 0.8)
- Overall match rate (percentage of URMA observations with aligned forecasts)
- Forecast hour distribution (showing data availability by lead time)
- Sample alignment checks with lead time information

## Usage Recommendation

For most use cases, start with `FORECAST_HOUR_AGGREGATION = 'separate'` to maximize training data while maintaining proper alignment. If overfitting becomes an issue, experiment with 'ensemble' or 'best_lead' methods.
