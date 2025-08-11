#!/usr/bin/env python
"""
XGBoost Regression Model for Weather Forecasting

This script trains an XGBoost model to predict maximum temperature observations
using reforecast meteorological data. It includes preprocessing, hyperparameter tuning,
feature engineering, and comprehensive model evaluation.
"""

import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed

import shap
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, KFold, ParameterSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer


class XGBoostForecastModel:
    """
    A class for building, training, and evaluating XGBoost models for weather forecasting.
    """
    
    def __init__(
        self, 
        station_id: str = "KBOI", 
        target_variable: str = "tmax_obs",
        csv_path: Optional[str] = None,
        exclude_vars: Optional[List[str]] = None,
        include_vars: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize the XGBoost forecast model.
        
        Args:
            station_id: Weather station identifier
            target_variable: Variable to predict
            csv_path: Path to the input CSV file
            exclude_vars: Variables to exclude from the model
            include_vars: Variables to include in the model
            random_state: Random seed for reproducibility
        """
        self.station_id = station_id
        self.target_variable = target_variable
        self.random_state = random_state
        
        # Set default paths and variables if not provided
        self.csv_path = csv_path or f"/content/{station_id}_2000_2009_f024.reforecast.csv"
        
        self.exclude_vars = exclude_vars or [
            'tmin_obs', 'hgt_ceiling', 'gust_sfc', 'cape_sfc', 'cin_sfc', 'tmin_2m'
        ]
        
        self.include_vars = include_vars or [
            "hgt_pres_850", "hgt_pres_925", "tmp_pres_700", "tmp_pres_850",
            "tmp_pres_925", "ugrd_pres_850", "ugrd_pres_925", "vgrd_pres_850",
            "vgrd_pres_925", "hgt_pres_700", "ugrd_pres_700", "vgrd_pres_700",
            "cape_sfc", "cin_sfc", "dlwrf_sfc", "dswrf_sfc", "gust_sfc",
            "hgt_ceiling", "lhtfl_sfc", "pres_sfc", "pres_msl", "shtfl_sfc",
            "soilw_bgrnd", "tcdc_eatm", "tmax_2m", "tmin_2m", "tmp_2m",
            "ulwrf_sfc", "uswrf_sfc", "ugrd_hgt", "vgrd_hgt"
        ]
        
        # Initialize data attributes
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.best_model = None
        self.results = {}

    def load_and_prepare_data(self) -> None:
        """
        Load data from CSV, prepare features, and split into train/val/test sets.
        """
        print(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        
        # Convert to datetime and add day of year feature
        self.df['valid_datetime'] = pd.to_datetime(self.df['valid_datetime'])
        self.df['doy'] = self.df['valid_datetime'].dt.dayofyear
        
        # Select columns to keep based on include and exclude lists
        keep_cols = [
            c for c in self.df.columns
            if (any(var in c for var in self.include_vars) and 
                not any(var in c for var in self.exclude_vars))
        ]
        
        # Add essential columns
        for c in ["valid_datetime", "doy", self.target_variable]:
            if c not in keep_cols:
                keep_cols.append(c)
        
        # Filter columns and set index
        self.df = self.df[keep_cols]
        self.df.set_index("valid_datetime", inplace=True)
        
        # Drop rows with missing values
        self.df.dropna(how='any', inplace=True)
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_variable])
        y = self.df[self.target_variable]
        
        # Split data: 70% train+val, 30% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Further split train+val into train (70%) and val (30%)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.3, random_state=self.random_state
        )
        
        # Report missing variables
        self._check_missing_variables()
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Feature count: {self.X_train.shape[1]}")

    def _check_missing_variables(self) -> None:
        """Check if any of the specified include variables are missing from the dataset."""
        df_base_cols = [col.split("_f024")[0] for col in self.df.columns]
        missing_vars = [var for var in self.include_vars if var not in df_base_cols]
        
        if missing_vars:
            print('Variables in INCLUDE_VARS not found in DataFrame columns:')
            print(missing_vars)
        else:
            print("All INCLUDE_VARS present in DataFrame columns.")

    def preprocess_data(self, X: pd.DataFrame, handle_missing: bool = True) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values.
        
        Args:
            X: Input features DataFrame
            handle_missing: Whether to handle missing values
            
        Returns:
            Processed DataFrame
        """
        X_processed = X.copy()
        
        if handle_missing:
            missing_stats = X_processed.isnull().sum()
            
            if missing_stats.sum() > 0:
                initial_rows = X_processed.shape[0]
                print(f"Found {missing_stats.sum()} missing values across {(missing_stats > 0).sum()} columns")
                
                X_processed = X_processed.dropna()
                
                rows_dropped = initial_rows - X_processed.shape[0]
                print(f"Dropped {rows_dropped} rows ({rows_dropped/initial_rows:.2%} of data) containing missing values")
        
        return X_processed

    def create_feature_interactions(
        self, 
        X: pd.DataFrame, 
        top_n: int = 5, 
        feature_importance: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Create interaction features between top important features.
        
        Args:
            X: Input features DataFrame
            top_n: Number of top features to consider for interactions
            feature_importance: Feature importance scores
            
        Returns:
            DataFrame with additional interaction features
        """
        X_enhanced = X.copy()
        
        if feature_importance is not None:
            top_features = feature_importance.sort_values(ascending=False).head(top_n).index.tolist()
            print(f"Creating interaction features from top {top_n} features: {top_features}")
            
            feature_count = 0
            for i in range(len(top_features)):
                for j in range(i+1, len(top_features)):
                    f1, f2 = top_features[i], top_features[j]
                    interaction_name = f'{f1}_x_{f2}'
                    X_enhanced[interaction_name] = X_enhanced[f1] * X_enhanced[f2]
                    feature_count += 1
            
            print(f"Added {feature_count} interaction features")
        
        return X_enhanced

    def run_xgb_cv(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        xgb_params: Dict[str, Any], 
        n_splits: int = 5, 
        candidate_num: Optional[int] = None, 
        total_candidates: Optional[int] = None
    ) -> Tuple[pd.Series, np.ndarray]:
        """
        Run XGBoost cross-validation with progress reporting.
        
        Args:
            X: Input features
            y: Target variable
            xgb_params: XGBoost parameters
            n_splits: Number of CV folds
            candidate_num: Current hyperparameter candidate number
            total_candidates: Total number of hyperparameter candidates
            
        Returns:
            True values and predictions from cross-validation
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        fold_data = [(fold_num, train_idx, test_idx) for fold_num, (train_idx, test_idx) in enumerate(kf.split(X))]
        
        def process_fold(fold_info):
            fold_num, train_idx, test_idx = fold_info
            fold_label = f"Candidate {candidate_num}/{total_candidates}, Fold {fold_num+1}/{n_splits}"
            print(f"[CV] {fold_label} training...")
            
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create a small validation set from training data for early stopping
            val_size = int(len(X_tr) * 0.1)
            X_tr_fit, X_tr_val = X_tr.iloc[:-val_size], X_tr.iloc[-val_size:]
            y_tr_fit, y_tr_val = y_tr.iloc[:-val_size], y_tr.iloc[-val_size:]
            
            model = XGBRegressor(**xgb_params)
            model.fit(
                X_tr_fit, y_tr_fit, 
                eval_set=[(X_tr_val, y_tr_val)], 
                verbose=False
            )
            
            y_te_pred = model.predict(X_te)
            r2 = r2_score(y_te, y_te_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_te_pred))
            
            print(f"[CV] {fold_label} done. R²={r2:.4f}, RMSE={rmse:.4f}")
            sys.stdout.flush()
            
            return y_te, y_te_pred
        
        # Determine number of parallel jobs to use
        n_jobs = xgb_params.get('n_jobs', 1)
        if n_jobs > 1:
            n_jobs_cv = min(n_jobs, n_splits)
            print(f"Running {n_splits} CV folds in parallel with {n_jobs_cv} jobs")
            results = Parallel(n_jobs=n_jobs_cv)(
                delayed(process_fold)(fold_info) for fold_info in fold_data
            )
        else:
            results = [process_fold(fold_info) for fold_info in fold_data]
        
        # Combine results from all folds
        cv_true, cv_pred = [], []
        for y_te, y_te_pred in results:
            cv_true.append(y_te)
            cv_pred.append(y_te_pred)
        
        cv_true = pd.concat(cv_true)
        cv_pred = np.concatenate(cv_pred)
        
        return cv_true, cv_pred

    def compute_feature_importance(
        self, 
        model: XGBRegressor, 
        X: pd.DataFrame, 
        importance_type: str = 'gain'
    ) -> pd.Series:
        """
        Compute feature importance using built-in XGBoost metrics.
        
        Args:
            model: Trained XGBoost model
            X: Input features
            importance_type: Type of importance ('gain' or 'weight')
            
        Returns:
            Series of feature importance scores
        """
        importance = model.get_booster().get_score(
            importance_type='total_gain' if importance_type == 'gain' else 'weight'
        )
        
        # Create a Series with all features, defaulting to 0 importance
        all_features = pd.Series(0, index=X.columns)
        existing_features = pd.Series(importance)
        all_features.update(existing_features)
        
        # Normalize to percentages
        all_features = 100 * all_features / all_features.sum()
        sorted_importance = all_features.sort_values(ascending=False)
        
        # Print top features
        print("Feature importances (gain):")
        for i, (feature, importance) in enumerate(sorted_importance.head(20).items()):
            print(f"{i+1:2d}. {feature}: {importance:.2f}%")
        
        return sorted_importance

    def compute_shap_importance(
        self, 
        model: XGBRegressor, 
        X_sample: pd.DataFrame
    ) -> Tuple[pd.Series, shap.TreeExplainer, np.ndarray, pd.DataFrame]:
        """
        Compute SHAP-based feature importance.
        
        Args:
            model: Trained XGBoost model
            X_sample: Sample of input features
            
        Returns:
            Tuple containing SHAP importance scores, explainer, SHAP values, and sampled data
        """
        print("Computing SHAP values (this may take a while for large datasets)...")
        
        # Sample data for SHAP analysis if needed
        X_shap = X_sample.sample(1000, random_state=self.random_state) if len(X_sample) > 1000 else X_sample
        if len(X_sample) > 1000:
            print(f"Using a sample of 1000 rows from {len(X_sample)} for SHAP analysis")
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        # Calculate and normalize importance scores
        mean_abs_shap = np.abs(shap_values).mean(0)
        shap_importance = pd.Series(mean_abs_shap, index=X_shap.columns)
        shap_importance = 100 * shap_importance / shap_importance.sum()
        sorted_importance = shap_importance.sort_values(ascending=False)
        
        # Print top features
        print("SHAP-based feature importances:")
        for i, (feature, importance) in enumerate(sorted_importance.head(20).items()):
            print(f"{i+1:2d}. {feature}: {importance:.2f}%")
        
        return sorted_importance, explainer, shap_values, X_shap

    def select_features_with_shap(
        self, 
        X: pd.DataFrame, 
        shap_importance: pd.Series, 
        threshold_percentile: int = 10
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features based on SHAP importance.
        
        Args:
            X: Input features
            shap_importance: SHAP importance scores
            threshold_percentile: Percentile threshold for feature selection
            
        Returns:
            Filtered DataFrame and list of kept features
        """
        threshold = np.percentile(shap_importance.values, threshold_percentile)
        keep_features = shap_importance[shap_importance > threshold].index.tolist()
        dropped_features = shap_importance[shap_importance <= threshold].index.tolist()
        
        print(f"SHAP importance drop threshold ({threshold_percentile}th percentile): {threshold:.5g}")
        print(f"Keeping {len(keep_features)} features, Dropping {len(dropped_features)} features")
        
        if dropped_features:
            print("Dropped features:")
            for i in range(0, len(dropped_features), 5):
                group = dropped_features[i:i+5]
                print(f"  {', '.join(group)}")
        
        return X[keep_features], keep_features

    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create a scikit-learn preprocessing pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Preprocessing pipeline
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features)],
            remainder='passthrough'
        )
        
        return preprocessor

    def train_model_with_progress(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        params: Dict[str, Any]
    ) -> Tuple[XGBRegressor, Tuple[List[float], List[float]]]:
        """
        Train an XGBoost model and track training progress.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: XGBoost parameters
            
        Returns:
            Tuple containing the trained model and training metrics
        """
        print("Training model with progress tracking...")
        
        model = XGBRegressor(**params)
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        
        # Extract training history
        evals_result = model.evals_result() if hasattr(model, 'evals_result') else {}
        
        if evals_result:
            train_rmse = evals_result['validation_0']['rmse']
            val_rmse = evals_result['validation_1']['rmse']
        else:
            print("evals_result() not available, manually tracking progress...")
            train_rmse, val_rmse = [], []
            n_trees = model.n_estimators
            
            for i in range(1, n_trees + 1):
                temp_model = XGBRegressor(**{**params, 'n_estimators': i})
                temp_model.fit(X_train, y_train)
                
                train_pred = temp_model.predict(X_train)
                val_pred = temp_model.predict(X_val)
                
                train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))
                val_rmse.append(np.sqrt(mean_squared_error(y_val, val_pred)))
                
                if i % 10 == 0:
                    print(f"Progress: {i}/{n_trees} trees")
        
        print(f"Training completed - {model.n_estimators} trees built")
        
        # Report best iteration
        if hasattr(model, 'best_ntree_limit'):
            print(f"Best iteration: {model.best_ntree_limit}")
        elif val_rmse:
            best_iter = np.argmin(val_rmse) + 1
            print(f"Best iteration (from history): {best_iter}")
        
        return model, (train_rmse, val_rmse)

    def print_metrics(
        self, 
        name: str, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate and print performance metrics.
        
        Args:
            name: Label for the metrics
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of metrics
        """
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"{name} R²:   {r2:.4f}")
        print(f"{name} MAE:  {mae:.4f}")
        print(f"{name} RMSE: {rmse:.4f}")
        
        return dict(r2=r2, mae=mae, rmse=rmse)

    def plot_training_progress(
        self, 
        train_rmse: List[float], 
        val_rmse: List[float], 
        metric: str = 'rmse', 
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot the training progress from stored evaluation history.
        
        Args:
            train_rmse: Training RMSE history
            val_rmse: Validation RMSE history
            metric: Metric name
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        iterations = range(1, len(train_rmse) + 1)
        
        ax.plot(iterations, train_rmse, label=f"Training {metric}", color='blue')
        ax.plot(iterations, val_rmse, label=f"Validation {metric}", color='red')
        
        best_iter = np.argmin(val_rmse) + 1
        ax.axvline(x=best_iter, color='r', linestyle='--', alpha=0.5, 
                   label=f'Best iteration: {best_iter}')
        
        ax.set_xlabel('Boosting Iterations')
        ax.set_ylabel(f'{metric.upper()}')
        ax.set_title(f'{metric.upper()} vs Iterations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_advanced_learning_curves(
        self, 
        train_rmse: List[float], 
        val_rmse: List[float], 
        figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """
        Create an advanced visualization of learning curves with both normal and log scales.
        
        Args:
            train_rmse: Training RMSE history
            val_rmse: Validation RMSE history
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        iterations = range(1, len(train_rmse) + 1)
        metric = 'RMSE'
        
        # Top left: Standard learning curve
        axes[0, 0].plot(iterations, train_rmse, label="Training", color='blue')
        axes[0, 0].plot(iterations, val_rmse, label="Validation", color='red')
        best_iter = np.argmin(val_rmse) + 1
        axes[0, 0].axvline(x=best_iter, color='k', linestyle='--', alpha=0.5, 
                          label=f'Best iteration: {best_iter}')
        axes[0, 0].set_xlabel('Boosting Iterations')
        axes[0, 0].set_ylabel(f'{metric}')
        axes[0, 0].set_title(f'{metric} Learning Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top right: Log scale learning curve
        axes[0, 1].plot(iterations, train_rmse, label="Training", color='blue')
        axes[0, 1].plot(iterations, val_rmse, label="Validation", color='red')
        axes[0, 1].axvline(x=best_iter, color='k', linestyle='--', alpha=0.5, 
                          label=f'Best iteration: {best_iter}')
        axes[0, 1].set_xlabel('Boosting Iterations')
        axes[0, 1].set_ylabel(f'{metric}')
        axes[0, 1].set_title(f'{metric} (Log Scale)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bottom left: Absolute gap between validation and training
        gap = np.array(val_rmse) - np.array(train_rmse)
        axes[1, 0].plot(iterations, gap, color='purple', label='Val - Train Gap')
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel('Boosting Iterations')
        axes[1, 0].set_ylabel(f'Absolute Gap ({metric})')
        axes[1, 0].set_title(f'Validation-Training Gap ({metric})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bottom right: Relative gap as percentage
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_gap = 100 * gap / np.array(train_rmse)
            rel_gap[~np.isfinite(rel_gap)] = 0
        
        axes[1, 1].plot(iterations, rel_gap, color='red', label='Relative Gap (%)')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Boosting Iterations')
        axes[1, 1].set_ylabel('Relative Gap (%)')
        axes[1, 1].set_title(f'Relative Validation-Training Gap')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def diagnostic_plot(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_test_pred: np.ndarray,
        cv_pred: np.ndarray,
        cv_true: pd.Series,
        feature_importance: pd.Series,
        title: str = "XGBoost Diagnostics"
    ) -> plt.Figure:
        """
        Create diagnostic plots for model evaluation.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            y_test_pred: Test predictions
            cv_pred: Cross-validation predictions
            cv_true: Cross-validation true values
            feature_importance: Feature importance scores
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        feature_names = X_train.columns
        min_val = min(np.min(y_test), np.min(y_test_pred), np.min(cv_true), np.min(cv_pred))
        max_val = max(np.max(y_test), np.max(y_test_pred), np.max(cv_true), np.max(cv_pred))
        
        fig = plt.figure(figsize=(18, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.3], height_ratios=[1, 1], figure=fig)
        
        axes = {}
        axes['tl'] = fig.add_subplot(gs[0, 0])
        axes['tm'] = fig.add_subplot(gs[0, 1])
        axes['bl'] = fig.add_subplot(gs[1, 0])
        axes['bm'] = fig.add_subplot(gs[1, 1])
        ax_feat = fig.add_subplot(gs[:, 2])
        
        # 1. Top left: Test set tmax_2m vs tmax_obs
        axes['tl'].scatter(X_test['tmax_2m'], y_test, alpha=0.5)
        axes['tl'].set_xlabel('tmax_2m (Feature)')
        axes['tl'].set_ylabel('tmax_obs (Target)')
        axes['tl'].set_title('tmax_2m vs Target (Test)')
        axes['tl'].grid(True)
        axes['tl'].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        r2_left = r2_score(y_test, X_test['tmax_2m'])
        mae_left = mean_absolute_error(y_test, X_test['tmax_2m'])
        rmse_left = np.sqrt(mean_squared_error(y_test, X_test['tmax_2m']))
        metrics_text_left = f"$R^2$ = {r2_left:.2f}\nMAE = {mae_left:.2f}\nRMSE = {rmse_left:.2f}"
        
        axes['tl'].annotate(metrics_text_left, xy=(0.05, 0.95), xycoords='axes fraction', 
                            fontsize=11, horizontalalignment='left', verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Top middle: Train set tmax_2m vs tmax_obs
        axes['tm'].scatter(X_train['tmax_2m'], y_train, alpha=0.5, color='green')
        axes['tm'].set_xlabel('tmax_2m (Feature)')
        axes['tm'].set_ylabel('tmax_obs (Target)')
        axes['tm'].set_title('tmax_2m vs Target (Train)')
        axes['tm'].grid(True)
        axes['tm'].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        r2_train = r2_score(y_train, X_train['tmax_2m'])
        mae_train = mean_absolute_error(y_train, X_train['tmax_2m'])
        rmse_train = np.sqrt(mean_squared_error(y_train, X_train['tmax_2m']))
        metrics_text_train = f"$R^2$ = {r2_train:.2f}\nMAE = {mae_train:.2f}\nRMSE = {rmse_train:.2f}"
        
        axes['tm'].annotate(metrics_text_train, xy=(0.05, 0.95), xycoords='axes fraction', 
                            fontsize=11, horizontalalignment='left', verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Bottom left: XGB CV Forecast vs tmax_obs (cross-validation)
        axes['bl'].scatter(cv_pred, cv_true, alpha=0.5, color='purple')
        axes['bl'].set_xlabel('XGB CV Forecast')
        axes['bl'].set_ylabel('tmax_obs (Target)')
        axes['bl'].set_title('XGB Cross-Validation Forecast vs Target')
        axes['bl'].grid(True)
        axes['bl'].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        cv_r2 = r2_score(cv_true, cv_pred)
        cv_mae = mean_absolute_error(cv_true, cv_pred)
        cv_rmse = np.sqrt(mean_squared_error(cv_true, cv_pred))
        metrics_text_cv = f"$R^2$ = {cv_r2:.2f}\nMAE = {cv_mae:.2f}\nRMSE = {cv_rmse:.2f}"
        
        axes['bl'].annotate(metrics_text_cv, xy=(0.05, 0.95), xycoords='axes fraction', 
                            fontsize=11, horizontalalignment='left', verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Bottom middle: XGB Forecast vs tmax_obs (Test set)
        axes['bm'].scatter(y_test_pred, y_test, alpha=0.5, color='orange')
        axes['bm'].set_xlabel('XGB Forecast')
        axes['bm'].set_ylabel('tmax_obs (Target)')
        axes['bm'].set_title('XGBoost Forecast vs Target (Test)')
        axes['bm'].grid(True)
        axes['bm'].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        metrics_text_test = f"$R^2$ = {r2_test:.2f}\nMAE = {mae_test:.2f}\nRMSE = {rmse_test:.2f}"
        
        axes['bm'].annotate(metrics_text_test, xy=(0.05, 0.95), xycoords='axes fraction', 
                            fontsize=11, horizontalalignment='left', verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Feature importance (right)
        if len(feature_importance) > 30:
            feature_importance = feature_importance.head(30)
        
        sorted_idx = np.argsort(feature_importance.values)
        ax_feat.barh(feature_importance.index[sorted_idx], feature_importance.values[sorted_idx], color='darkblue')
        ax_feat.set_xlabel('Importance (%)')
        ax_feat.set_title('XGBoost Feature Importance')
        ax_feat.grid(axis='x', linestyle='--', alpha=0.7)
        
        fig.suptitle(title, fontsize=20)
        plt.show()
        
        return fig

    def create_shap_plots(
        self, 
        explainer: shap.TreeExplainer, 
        shap_values: np.ndarray, 
        X_shap: pd.DataFrame
    ) -> None:
        """
        Create SHAP interpretation plots.
        
        Args:
            explainer: SHAP explainer
            shap_values: SHAP values
            X_shap: Sample data for SHAP analysis
        """
        # Summary bar plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Summary dot plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.title("SHAP Feature Impact Distribution", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Dependence plots for top features
        shap_sum = np.abs(shap_values).mean(0)
        top_inds = np.argsort(-shap_sum)[:3]
        top_features = X_shap.columns[top_inds]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, feature in enumerate(top_features):
            shap.dependence_plot(
                feature,
                shap_values,
                X_shap,
                interaction_index='auto',
                ax=axes[i],
                show=False
            )
            axes[i].set_title(f"SHAP Dependence: {feature}", fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        # Force plots for sample instances
        sample_indices = np.random.choice(len(X_shap), min(3, len(X_shap)), replace=False)
        shap.initjs()
        
        for idx in sample_indices:
            plt.figure(figsize=(14, 3))
            shap.force_plot(
                explainer.expected_value,
                shap_values[idx],
                X_shap.iloc[idx],
                matplotlib=True,
                show=False
            )
            plt.title(f"SHAP Force Plot - Sample {idx}", fontsize=12)
            plt.tight_layout()
            plt.show()

    def run_hyperparameter_tuning(
        self, 
        debug: bool = False, 
        max_fits: int = 25, 
        n_folds: int = 5, 
        n_jobs: int = 4
    ) -> Dict[str, Any]:
        """
        Run the complete XGBoost training pipeline with hyperparameter tuning.
        
        Args:
            debug: Whether to run in debug mode with fewer iterations
            max_fits: Maximum number of hyperparameter combinations to try
            n_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary of results
        """
        print("\n===== Starting XGBoost training pipeline with improvements =====")
        start_time = time.time()
        
        print("\n----- Data Preprocessing -----")
        X_train_proc = self.preprocess_data(self.X_train)
        X_val_proc = self.preprocess_data(self.X_val)
        X_test_proc = self.preprocess_data(self.X_test)
        
        print("\n----- Hyperparameter Tuning with Early Stopping -----")
        param_grid = {
            'max_depth': [3, 5, 7, 10, 12],
            'min_child_weight': [3, 5, 7, 10],
            'gamma': [0, 0.1, 0.3, 0.5],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 2, 5, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'early_stopping_rounds': [5]
        }
        
        # Check if GPU is available
        gpu_available = False
        try:
            # Try to import necessary packages to check GPU availability
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            try:
                # Alternative check if torch is not available
                import subprocess
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                gpu_available = result.returncode == 0
            except (ImportError, FileNotFoundError):
                gpu_available = False
        
        # Set base parameters based on GPU availability
        base_params = dict(
            n_estimators=40,
            random_state=self.random_state,
            n_jobs=n_jobs,
            eval_metric="rmse"
        )
        
        # Add GPU-specific parameters if GPU is available
        if gpu_available:
            print("GPU detected, enabling GPU acceleration for XGBoost")
            base_params.update({
                'tree_method': 'hist',
                'device': 'cuda'
            })
        else:
            print("No GPU detected, using CPU for XGBoost")
        
        # Simplify for debug mode
        if debug:
            param_grid = {k: v[:2] for k, v in param_grid.items()}
            max_fits = 3
        
        # Sample hyperparameter combinations
        param_list = list(ParameterSampler(param_grid, n_iter=max_fits, random_state=self.random_state))
        cv_scores, params_used = [], []
        
        # Evaluate each hyperparameter combination
        for cand_idx, params in enumerate(param_list):
            print(f"\n[Candidate {cand_idx+1}/{len(param_list)}] Params: {params}")
            all_params = {**base_params, **params}
            
            cv_true, cv_pred = self.run_xgb_cv(
                X_train_proc, self.y_train, all_params, n_splits=n_folds,
                candidate_num=cand_idx+1, total_candidates=len(param_list)
            )
            
            rmse = np.sqrt(mean_squared_error(cv_true, cv_pred))
            print(f"[Candidate {cand_idx+1}/{len(param_list)}] Mean CV RMSE: {rmse:.4f}")
            
            cv_scores.append(rmse)
            params_used.append(all_params)
            
            if debug and cand_idx == 0:
                print("Debug mode: stopping after first candidate.")
                break
        
        # Select best hyperparameters
        best_idx = int(np.argmin(cv_scores))
        best_params = params_used[best_idx]
        
        print("\nBest candidate:", best_params)
        print(f"Best CV RMSE: {cv_scores[best_idx]:.4f}")
        
        print("\n----- Training Best Model with Progress Tracking -----")
        model, (train_rmse, val_rmse) = self.train_model_with_progress(
            X_train_proc, self.y_train, X_val_proc, self.y_val, best_params
        )
        
        print("\n----- Training Progress Visualization -----")
        self.plot_training_progress(train_rmse, val_rmse)
        
        print("\n----- Advanced Learning Curve Analysis -----")
        self.plot_advanced_learning_curves(train_rmse, val_rmse)
        
        print("\n----- Feature Importance Analysis -----")
        feature_importance = self.compute_feature_importance(model, X_train_proc)
        shap_importance, explainer, shap_values, X_shap = self.compute_shap_importance(model, X_train_proc)
        
        print("\n----- Feature Selection using SHAP values -----")
        X_train_fs, keep_features = self.select_features_with_shap(X_train_proc, shap_importance, threshold_percentile=10)
        X_val_fs = X_val_proc[keep_features]
        X_test_fs = X_test_proc[keep_features]
        
        print("\n----- Feature Engineering: Creating Interaction Features -----")
        X_train_fe = self.create_feature_interactions(X_train_fs, top_n=5, feature_importance=shap_importance[keep_features])
        X_val_fe = self.create_feature_interactions(X_val_fs, top_n=5, feature_importance=shap_importance[keep_features])
        X_test_fe = self.create_feature_interactions(X_test_fs, top_n=5, feature_importance=shap_importance[keep_features])
        
        print("\n----- Training Final Model with Feature Engineering -----")
        model_fe, (train_rmse_fe, val_rmse_fe) = self.train_model_with_progress(
            X_train_fe, self.y_train, X_val_fe, self.y_val, best_params
        )
        
        print("\n----- Final Model Training Progress -----")
        self.plot_training_progress(train_rmse_fe, val_rmse_fe)
        
        print("\n----- Final Model Learning Curve Analysis -----")
        self.plot_advanced_learning_curves(train_rmse_fe, val_rmse_fe)
        
        feature_importance_fe = self.compute_feature_importance(model_fe, X_train_fe)
        shap_importance_fe, explainer_fe, shap_values_fe, X_shap_fe = self.compute_shap_importance(model_fe, X_train_fe)
        
        print("\n----- Cross-validation with Feature Engineering -----")
        cv_true_fe, cv_pred_fe = self.run_xgb_cv(
            X_train_fe, self.y_train, best_params, n_splits=n_folds,
            candidate_num=1, total_candidates=1
        )
        
        print("\n----- Generating Final Predictions -----")
        y_val_pred_fe = model_fe.predict(X_val_fe)
        y_test_pred_fe = model_fe.predict(X_test_fe)
        
        print("\n----- Final Model Evaluation -----")
        val_stats_fe = self.print_metrics("Validation", self.y_val, y_val_pred_fe)
        test_stats_fe = self.print_metrics("Test", self.y_test, y_test_pred_fe)
        
        print("\n----- Creating Diagnostic Visualizations -----")
        fig = self.diagnostic_plot(
            X_train_fe, self.y_train,
            X_test_fe, self.y_test,
            y_test_pred_fe,
            cv_pred_fe, cv_true_fe,
            feature_importance_fe,
            title="XGBoost with Feature Engineering & SHAP-based Selection"
        )
        
        self.create_shap_plots(explainer_fe, shap_values_fe, X_shap_fe)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")
        
        # Store results
        self.best_model = model_fe
        self.results = {
            "best_model": model_fe,
            "best_params": best_params,
            "cv_rmse": cv_scores[best_idx],
            "feature_importance": feature_importance_fe,
            "shap_importance": shap_importance_fe,
            "validation_stats": val_stats_fe,
            "test_stats": test_stats_fe,
            "selected_features": list(X_train_fe.columns),
            "cv_true": cv_true_fe,
            "cv_pred": cv_pred_fe,
            "y_val_pred": y_val_pred_fe,
            "y_test_pred": y_test_pred_fe,
            "explainer": explainer_fe,
            "shap_values": shap_values_fe,
            "execution_time": execution_time,
            "training_history": (train_rmse_fe, val_rmse_fe)
        }
        
        return self.results


def main():
    """Main function to run the XGBoost model pipeline."""
    # Initialize model with default settings
    model = XGBoostForecastModel(
        station_id="KBOI",
        target_variable="tmax_obs",
        csv_path=None  # Will use default path based on station_id
    )
    
    # Load and prepare data
    model.load_and_prepare_data()
    
    # Run hyperparameter tuning and complete model pipeline
    results = model.run_hyperparameter_tuning(
        debug=False,  # Set to True for a quick test run
        max_fits=5,   # Number of hyperparameter combinations to try
        n_folds=5,    # Number of CV folds
        n_jobs=5      # Parallel processing
    )
    
    # Print final results summary
    best_model = results["best_model"]
    best_params = results["best_params"]
    test_stats = results["test_stats"]
    
    print(f"\nBest model test RMSE: {test_stats['rmse']:.4f}")
    print(f"Best model test R²: {test_stats['r2']:.4f}")
    print(f"Top 5 features:")
    for i, (feature, importance) in enumerate(results["feature_importance"].head(5).items()):
        print(f"  {i+1}. {feature}: {importance:.2f}%")

    return results

if __name__ == "__main__":

    results = main()
    print("\nXGBoost model pipeline completed successfully.")