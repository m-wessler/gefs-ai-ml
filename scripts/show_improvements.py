#!/usr/bin/env python3

"""
Quick verification script to show the data alignment improvements.
"""

# Show the key improvements made
print("=== GEFS ML DATA ALIGNMENT IMPROVEMENTS ===\n")

print("1. IMPROVED TIME MATCHING:")
print("   - Before: Loose pairing could mismatch GEFS f024 with NBM f048")
print("   - After: Exact forecast time alignment required")
print("   - Added: forecast_lead_hours tracking for diagnostics")

print("\n2. ENHANCED VERIFICATION:")
print("   - Before: Overall GEFS-NBM correlation only")
print("   - After: Lead time-specific correlation checking")
print("   - Added: Forecast hour distribution reporting")

print("\n3. FORECAST HOUR AGGREGATION:")
print("   - separate: Keep each forecast hour as separate training examples (default)")
print("   - ensemble: Average multiple forecast hours for same verification")
print("   - best_lead: Use only forecast closest to 24h lead time")

print("\n4. CONFIGURATION:")
print("   FORECAST_HOUR_AGGREGATION = 'separate'  # Current setting")

print("\n5. EXPECTED IMPROVEMENTS:")
print("   - Better NBM verification scatterplots")
print("   - Improved model performance")
print("   - Better diagnostic capabilities")

print("\n=== KEY CHANGES TO ADDRESS MULTI-FORECAST AGGREGATION ===")

print("\nOLD create_time_matched_dataset():")
print("   - Could pair mismatched GEFS/NBM forecasts")
print("   - No lead time information")
print("   - Poor alignment diagnostics")

print("\nNEW create_time_matched_dataset():")
print("   - Requires exact GEFS-NBM time match")
print("   - Tracks forecast lead hours")
print("   - Reports match statistics")

print("\nThe improved alignment should resolve the NBM verification scatterplot issues")
print("that occurred when aggregating across forecast hours.\n")

# Check if the latest model was trained with improvements
import json
from pathlib import Path

model_path = Path("N:/projects/michael.wessler/gefs-ai-ml/models")
metadata_file = model_path / "gefs_ml_metadata_tmax_latest.json"

if metadata_file.exists():
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print("=== LATEST MODEL METADATA ===")
    
    config = metadata.get('training_config', {})
    print(f"Forecast hours: {config.get('FORECAST_HOURS', 'N/A')}")
    print(f"Forecast hour aggregation: {config.get('FORECAST_HOUR_AGGREGATION', 'Not specified (using old version)')}")
    
    perf = metadata.get('performance_metrics', {})
    if perf:
        print("\nPerformance metrics available for:")
        for key in perf.keys():
            print(f"  - {key}")
    
    training_time = metadata.get('model_info', {}).get('training_timestamp', 'Unknown')
    print(f"\nModel trained: {training_time}")
    
    if 'FORECAST_HOUR_AGGREGATION' in config:
        print("\n✅ Model was trained with improved alignment logic!")
    else:
        print("\n⚠️  Model was trained with old alignment logic")
else:
    print("No latest model metadata found")
