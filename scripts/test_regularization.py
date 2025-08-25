#!/usr/bin/env python3

"""Quick test to verify the regularization parameters work."""

from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Generate some test data
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

# Test regularized model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=1  # Use single job for testing
)

try:
    model.fit(X, y)
    print(f"✓ Model training successful")
    print(f"OOB Score: {model.oob_score_:.3f}")
    print(f"Feature importances shape: {model.feature_importances_.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
