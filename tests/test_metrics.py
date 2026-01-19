"""
Tests for the metrics module.
"""

import numpy as np
import pytest

from raman_bench.metrics import ClassificationMetrics, RegressionMetrics, compute_metrics


class TestClassificationMetrics:
    """Tests for classification metrics."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 0, 1, 0, 2, 2, 0, 1, 1, 0])
        y_proba = np.random.rand(10, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        return y_true, y_pred, y_proba

    def test_compute_all_metrics(self, sample_data):
        """Test computing all classification metrics."""
        y_true, y_pred, y_proba = sample_data
        calculator = ClassificationMetrics()

        metrics = calculator.compute_all(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_accuracy(self, sample_data):
        """Test accuracy computation."""
        y_true, y_pred, _ = sample_data
        calculator = ClassificationMetrics()

        accuracy = calculator.accuracy(y_true, y_pred)
        assert 0 <= accuracy <= 1

    def test_confusion_matrix(self, sample_data):
        """Test confusion matrix computation."""
        y_true, y_pred, _ = sample_data
        calculator = ClassificationMetrics()

        cm = calculator.confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm.sum() == len(y_true)


class TestRegressionMetrics:
    """Tests for regression metrics."""

    @pytest.fixture
    def sample_data(self):
        """Create sample regression data."""
        np.random.seed(42)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.2])
        return y_true, y_pred

    def test_compute_all_metrics(self, sample_data):
        """Test computing all regression metrics."""
        y_true, y_pred = sample_data
        calculator = RegressionMetrics()

        metrics = calculator.compute_all(y_true, y_pred)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mse"] >= 0
        assert metrics["r2"] <= 1

    def test_mse(self, sample_data):
        """Test MSE computation."""
        y_true, y_pred = sample_data
        calculator = RegressionMetrics()

        mse = calculator.mse(y_true, y_pred)
        assert mse >= 0

    def test_r2(self, sample_data):
        """Test R² computation."""
        y_true, y_pred = sample_data
        calculator = RegressionMetrics()

        r2 = calculator.r2(y_true, y_pred)
        assert r2 <= 1

    def test_residuals(self, sample_data):
        """Test residual analysis."""
        y_true, y_pred = sample_data
        calculator = RegressionMetrics()

        residuals = calculator.get_residuals(y_true, y_pred)

        assert "residuals" in residuals
        assert "mean" in residuals
        assert "std" in residuals
        assert len(residuals["residuals"]) == len(y_true)


class TestComputeMetrics:
    """Tests for the compute_metrics utility function."""

    def test_classification_metrics(self):
        """Test compute_metrics for classification."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        metrics = compute_metrics(y_true, y_pred, task_type="classification")

        assert "accuracy" in metrics
        assert "f1_score" in metrics

    def test_regression_metrics(self):
        """Test compute_metrics for regression."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.0, 3.2])

        metrics = compute_metrics(y_true, y_pred, task_type="regression")

        assert "mse" in metrics
        assert "r2" in metrics

    def test_invalid_task_type(self):
        """Test that invalid task type raises error."""
        with pytest.raises(ValueError):
            compute_metrics(
                np.array([0, 1]),
                np.array([0, 1]),
                task_type="invalid",
            )

