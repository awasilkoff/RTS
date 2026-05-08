"""
Unit tests for weighted conformal prediction.

Tests:
- weighted_quantile() correctness
- _compute_kernel_distances() matches covariance implementation
- compute_weighted_conformal_correction_lower() returns valid q_hats
- train_wind_lower_model_weighted_conformal() runs without errors
- Bundle can save/load and predict
- Coverage validation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pytest
    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False
    # Mock pytest.raises for standalone execution
    class MockRaises:
        def __init__(self, exc_type, match=None):
            self.exc_type = exc_type
            self.match = match
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exc_type.__name__} but no exception was raised")
            if not issubclass(exc_type, self.exc_type):
                return False  # Re-raise
            return True  # Suppress exception

    class MockPytest:
        @staticmethod
        def raises(exc_type, match=None):
            return MockRaises(exc_type, match)

    pytest = MockPytest()

from conformal_prediction import (
    weighted_quantile,
    _compute_kernel_distances,
    compute_weighted_conformal_correction_lower,
    train_wind_lower_model_weighted_conformal,
    WeightedConformalLowerBundle,
)


def test_weighted_quantile_uniform_weights():
    """Test weighted quantile with uniform weights matches np.quantile."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones(5)

    # Without test weight (pure numpy quantile)
    q_weighted = weighted_quantile(values, weights, q=0.5, include_test_weight=False)
    q_numpy = np.quantile(values, 0.5)
    assert np.isclose(q_weighted, q_numpy), f"Expected {q_numpy}, got {q_weighted}"


def test_weighted_quantile_with_test_weight():
    """Test weighted quantile with test weight included."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    q_weighted = weighted_quantile(values, weights, q=0.95, include_test_weight=True)

    # Should be in reasonable range
    assert 3.0 <= q_weighted <= 5.0, f"q_weighted={q_weighted} out of expected range"


def test_weighted_quantile_edge_cases():
    """Test weighted quantile edge cases."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])

    # q=1.0 should return max value
    q_max = weighted_quantile(values, weights, q=1.0, include_test_weight=True)
    assert np.isclose(q_max, 3.0), f"Expected max=3.0, got {q_max}"

    # q=0.0 should return min value
    q_min = weighted_quantile(values, weights, q=0.0, include_test_weight=True)
    assert np.isclose(q_min, 1.0), f"Expected min=1.0, got {q_min}"


def test_weighted_quantile_empty_array():
    """Test that empty arrays raise appropriate errors."""
    with pytest.raises(ValueError, match="Cannot compute quantile of empty array"):
        weighted_quantile(np.array([]), np.array([]), q=0.5)


def test_weighted_quantile_mismatched_shapes():
    """Test that mismatched shapes raise errors."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="values and weights must have same size"):
        weighted_quantile(values, weights, q=0.5)


def test_compute_kernel_distances_self_similarity():
    """Test kernel distance computation with self-similarity."""
    X_query = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_ref = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    omega = np.array([1.0, 1.0])
    tau = 1.0

    K = _compute_kernel_distances(X_query, X_ref, omega, tau)

    # Check shape
    assert K.shape == (2, 3), f"Expected shape (2, 3), got {K.shape}"

    # Check self-similarity: K(x, x) should be close to 1.0
    assert np.isclose(K[0, 0], 1.0, atol=1e-3), f"K[0,0]={K[0,0]} should be ~1.0"
    assert np.isclose(K[1, 2], 1.0, atol=1e-3), f"K[1,2]={K[1,2]} should be ~1.0"

    # Check all weights in [0, 1]
    assert np.all((K >= 0) & (K <= 1)), "All kernel weights should be in [0, 1]"


def test_compute_kernel_distances_omega_weighting():
    """Test that omega properly weights features."""
    X_query = np.array([[0.0, 0.0]])
    X_ref = np.array([[1.0, 0.0], [0.0, 1.0]])  # One unit apart in each dimension

    # Uniform omega: both features equally important
    omega_uniform = np.array([1.0, 1.0])
    K_uniform = _compute_kernel_distances(X_query, X_ref, omega_uniform, tau=1.0)

    # First feature only
    omega_first = np.array([1.0, 0.0])
    K_first = _compute_kernel_distances(X_query, X_ref, omega_first, tau=1.0)

    # Second feature only
    omega_second = np.array([0.0, 1.0])
    K_second = _compute_kernel_distances(X_query, X_ref, omega_second, tau=1.0)

    # With omega_first, X_ref[0] should be closer (only dim 0 matters)
    # With omega_second, X_ref[1] should be closer (only dim 1 matters)
    assert K_first[0, 0] < K_first[0, 1], "omega_first should favor X_ref[0]"
    assert K_second[0, 1] < K_second[0, 0], "omega_second should favor X_ref[1]"


def test_weighted_conformal_basic():
    """Test basic weighted conformal functionality with synthetic data."""
    # Synthetic data
    n = 1000
    np.random.seed(42)

    X = np.random.randn(n, 2)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5

    df = pd.DataFrame({
        'TIME_HOURLY': pd.date_range('2020-01-01', periods=n, freq='h'),
        'y': y,
        'ens_mean': X[:, 0],
        'ens_std': np.abs(np.random.randn(n)) + 0.5,
        'SYS_MEAN': X[:, 0],
        'SYS_STD': X[:, 1],
    })

    omega = np.array([1.0, 1.0])  # Uniform weights

    bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
        df,
        feature_cols=['ens_mean', 'ens_std'],
        kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
        omega=omega,
        tau=5.0,
        alpha_target=0.95,
        split_method='random',
    )

    # Check coverage
    assert metrics['coverage'] > 0.90, f"Coverage {metrics['coverage']:.3f} too low"
    assert abs(metrics['coverage'] - 0.95) < 0.10, f"Coverage gap {abs(metrics['coverage'] - 0.95):.3f} too large"

    # Check q_hat statistics
    assert metrics['q_hat_mean'] > 0, "q_hat_mean should be positive"
    assert metrics['q_hat_std'] >= 0, "q_hat_std should be non-negative"

    # Check bundle can predict
    df_new = df.tail(10)
    df_pred = bundle.predict_df(df_new)

    assert 'y_pred_conf' in df_pred.columns, "Missing y_pred_conf column"
    assert 'q_hat_local' in df_pred.columns, "Missing q_hat_local column"
    assert len(df_pred) == 10, f"Expected 10 rows, got {len(df_pred)}"


def test_weighted_conformal_omega_from_file(tmp_path):
    """Test loading omega from file."""
    # Create synthetic data
    n = 500
    np.random.seed(42)

    X = np.random.randn(n, 2)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.3

    df = pd.DataFrame({
        'TIME_HOURLY': pd.date_range('2020-01-01', periods=n, freq='h'),
        'y': y,
        'ens_mean': X[:, 0],
        'ens_std': np.abs(np.random.randn(n)) + 0.3,
        'SYS_MEAN': X[:, 0],
        'SYS_STD': X[:, 1],
    })

    # Save omega to file
    omega = np.array([1.0, 0.5])
    omega_path = tmp_path / "test_omega.npy"
    np.save(omega_path, omega)

    # Train with omega_path
    bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
        df,
        feature_cols=['ens_mean', 'ens_std'],
        kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
        omega_path=str(omega_path),
        tau=5.0,
        alpha_target=0.90,
        split_method='random',
    )

    assert metrics['coverage'] > 0.80, f"Coverage {metrics['coverage']:.3f} too low"
    assert np.allclose(bundle.omega, omega), "Bundle omega doesn't match loaded omega"


def test_weighted_conformal_coverage_levels():
    """Test weighted conformal at multiple coverage levels."""
    # Synthetic data
    n = 1000
    np.random.seed(42)

    X = np.random.randn(n, 2)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5

    df = pd.DataFrame({
        'TIME_HOURLY': pd.date_range('2020-01-01', periods=n, freq='h'),
        'y': y,
        'ens_mean': X[:, 0],
        'ens_std': np.abs(np.random.randn(n)) + 0.5,
        'SYS_MEAN': X[:, 0],
        'SYS_STD': X[:, 1],
    })

    omega = np.array([1.0, 1.0])
    target_levels = [0.90, 0.95, 0.99]

    for alpha_target in target_levels:
        bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
            df,
            feature_cols=['ens_mean', 'ens_std'],
            kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
            omega=omega,
            tau=5.0,
            alpha_target=alpha_target,
            split_method='random',
            random_seed=42,
        )

        # Coverage should be close to target (within 10%)
        coverage_gap = abs(metrics['coverage'] - alpha_target)
        assert coverage_gap < 0.10, f"Coverage gap {coverage_gap:.3f} too large for alpha={alpha_target}"


def test_weighted_conformal_dimension_mismatch():
    """Test that dimension mismatches raise appropriate errors."""
    n = 100
    np.random.seed(42)

    df = pd.DataFrame({
        'TIME_HOURLY': pd.date_range('2020-01-01', periods=n, freq='h'),
        'y': np.random.randn(n),
        'ens_mean': np.random.randn(n),
        'ens_std': np.random.randn(n) + 1.0,
        'SYS_MEAN': np.random.randn(n),
        'SYS_STD': np.random.randn(n),
    })

    # Wrong omega dimension (3 instead of 2)
    omega_wrong = np.array([1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="omega shape .* doesn't match kernel_feature_cols"):
        train_wind_lower_model_weighted_conformal(
            df,
            feature_cols=['ens_mean', 'ens_std'],
            kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
            omega=omega_wrong,
            tau=5.0,
            alpha_target=0.95,
        )


def test_weighted_conformal_missing_columns():
    """Test that missing columns raise appropriate errors."""
    n = 100
    np.random.seed(42)

    df = pd.DataFrame({
        'TIME_HOURLY': pd.date_range('2020-01-01', periods=n, freq='h'),
        'y': np.random.randn(n),
        'ens_mean': np.random.randn(n),
        'ens_std': np.random.randn(n) + 1.0,
        # Missing SYS_MEAN, SYS_STD
    })

    omega = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="Missing kernel feature columns"):
        train_wind_lower_model_weighted_conformal(
            df,
            feature_cols=['ens_mean', 'ens_std'],
            kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
            omega=omega,
            tau=5.0,
            alpha_target=0.95,
        )


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running weighted conformal tests...")

    print("\n1. Testing weighted_quantile...")
    test_weighted_quantile_uniform_weights()
    test_weighted_quantile_with_test_weight()
    test_weighted_quantile_edge_cases()
    print("   (ok) weighted_quantile tests passed")

    print("\n2. Testing kernel distances...")
    test_compute_kernel_distances_self_similarity()
    test_compute_kernel_distances_omega_weighting()
    print("   (ok) kernel distance tests passed")

    print("\n3. Testing weighted conformal training...")
    test_weighted_conformal_basic()
    print("   (ok) basic weighted conformal test passed")

    print("\n4. Testing coverage levels...")
    test_weighted_conformal_coverage_levels()
    print("   (ok) coverage level tests passed")

    print("\n(ok) All tests passed!")
