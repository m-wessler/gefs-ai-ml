#!/usr/bin/env python3

"""
Inspect Trained GEFS ML Model

This script allows you to inspect a trained model's configuration, features, 
and performance metrics without retraining.

Usage:
    python inspect_model.py [target_type]
    
    target_type: 'tmax' or 'tmin' (default: 'tmax')
"""

import json
import sys
from pathlib import Path

# Import the model loading function
from gefs_ml_trainer import load_model_and_metadata

def print_model_summary(metadata):
    """Print a comprehensive summary of the model."""
    
    print("=" * 80)
    print("GEFS ML MODEL INSPECTION REPORT")
    print("=" * 80)
    
    # Model basic info
    model_info = metadata['model_info']
    print(f"\n📊 MODEL INFORMATION")
    print(f"   Target Variable: {model_info['target_description']} ({model_info['target_type']})")
    print(f"   Model Type: {model_info['model_type']}")
    print(f"   Training Date: {model_info['training_timestamp']}")
    print(f"   Script Version: {model_info.get('script_version', 'Unknown')}")
    
    # Training configuration
    config = metadata['training_config']
    print(f"\n⚙️  TRAINING CONFIGURATION")
    print(f"   Forecast Hours: {config['FORECAST_HOURS']}")
    print(f"   QC Threshold: {config['QC_THRESHOLD']}°C")
    print(f"   Hyperparameter Tuning: {config['USE_HYPERPARAMETER_TUNING']}")
    print(f"   Ensemble Models: {config['USE_ENSEMBLE_MODELS']}")
    print(f"   Feature Selection: {config['USE_FEATURE_SELECTION']}")
    print(f"   NBM as Predictor: {config['INCLUDE_NBM_AS_PREDICTOR']}")
    print(f"   GEFS Target as Predictor: {config['INCLUDE_GEFS_AS_PREDICTOR']}")
    print(f"   All Available Stations: {config['USE_ALL_AVAILABLE_STATIONS']}")
    if config['USE_ALL_AVAILABLE_STATIONS']:
        print(f"   Max Stations: {config['MAX_STATIONS']}")
        print(f"   Random Seed: {config['RANDOM_STATION_SEED']}")
    
    # Feature information
    features = metadata['features']
    print(f"\n🔧 FEATURE INFORMATION")
    print(f"   Total Features Used: {features['n_features']}")
    
    feature_types = features['feature_types']
    print(f"   GEFS Atmospheric: {len(feature_types['gefs_atmospheric'])}")
    print(f"   Engineered Features: {len(feature_types['engineered'])}")
    print(f"   NBM Features: {len(feature_types['nbm_features'])}")
    print(f"   Other Features: {len(feature_types['other'])}")
    
    # Performance metrics
    performance = metadata['performance_metrics']
    print(f"\n📈 MODEL PERFORMANCE")
    print(f"   {'Split':<8} {'Model':<12} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Samples':<8}")
    print(f"   {'-'*60}")
    
    for split in ['train', 'val', 'test']:
        for model_type in ['ML', 'NBM']:
            key = f'{model_type}_{split}'
            if key in performance:
                metrics = performance[key]
                print(f"   {split:<8} {model_type:<12} {metrics['mae']:<8.2f} {metrics['rmse']:<8.2f} "
                      f"{metrics['r2']:<8.3f} {metrics['n_samples']:<8,}")
    
    # QC statistics
    qc_stats = metadata['qc_stats']
    print(f"\n🔍 QUALITY CONTROL")
    if 'error' in qc_stats:
        print(f"   Status: Error - {qc_stats['error']}")
    elif 'total_records' in qc_stats:
        print(f"   Total Records: {qc_stats['total_records']:,}")
        print(f"   Records Passing QC: {qc_stats['qc_passed']:,}")
        print(f"   QC Pass Rate: {qc_stats['qc_pass_rate']*100:.1f}%")
        print(f"   QC Threshold: {qc_stats['threshold_used']}°C")
    
    # Data preprocessing info
    preprocessing = metadata['data_preprocessing']
    print(f"\n🔄 DATA PREPROCESSING")
    print(f"   URMA Temperature Conversion: {preprocessing['urma_temp_conversion']}")
    print(f"   Prohibited Features: {preprocessing['prohibited_features']['target_specific']}")
    
    return True

def print_detailed_features(metadata):
    """Print detailed feature information."""
    
    features = metadata['features']
    feature_types = features['feature_types']
    
    print(f"\n" + "="*80)
    print("DETAILED FEATURE BREAKDOWN")
    print("="*80)
    
    print(f"\n🌤️  GEFS ATMOSPHERIC VARIABLES ({len(feature_types['gefs_atmospheric'])})")
    for i, feature in enumerate(feature_types['gefs_atmospheric'], 1):
        print(f"   {i:2d}. {feature}")
    
    print(f"\n⚡ ENGINEERED FEATURES ({len(feature_types['engineered'])})")
    for i, feature in enumerate(feature_types['engineered'], 1):
        print(f"   {i:2d}. {feature}")
    
    if feature_types['nbm_features']:
        print(f"\n📡 NBM FEATURES ({len(feature_types['nbm_features'])})")
        for i, feature in enumerate(feature_types['nbm_features'], 1):
            print(f"   {i:2d}. {feature}")
    
    if feature_types['other']:
        print(f"\n❓ OTHER FEATURES ({len(feature_types['other'])})")
        for i, feature in enumerate(feature_types['other'], 1):
            print(f"   {i:2d}. {feature}")

def main():
    """Main inspection function."""
    
    # Get target type from command line or use default
    target_type = sys.argv[1] if len(sys.argv) > 1 else 'tmax'
    
    if target_type not in ['tmax', 'tmin']:
        print("Error: target_type must be 'tmax' or 'tmin'")
        sys.exit(1)
    
    try:
        # Load model metadata
        print(f"Loading model metadata for {target_type}...")
        model, metadata = load_model_and_metadata(target_type=target_type)
        
        # Print summary
        print_model_summary(metadata)
        
        # Ask if user wants detailed features
        print(f"\n" + "="*80)
        response = input("Show detailed feature breakdown? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print_detailed_features(metadata)
        
        # Show model files
        models_dir = Path('models')
        if models_dir.exists():
            print(f"\n📁 AVAILABLE MODEL FILES")
            model_files = list(models_dir.glob(f"*{target_type}*"))
            for file in sorted(model_files):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   {file.name} ({size_mb:.1f} MB)")
        
        print(f"\n✅ Model inspection completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: No trained model found for {target_type}")
        print(f"   Please run gefs_ml_trainer.py first to train a model.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        raise

if __name__ == "__main__":
    main()
