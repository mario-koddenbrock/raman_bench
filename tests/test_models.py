"""
Tests for the models module.
"""

import numpy as np
import pytest

from raman_bench.models import (
    RandomForestModel,
    SVMModel,
    RandomForestRegressor,
    get_model,
    list_models,
)


class TestClassificationModels:
    """Tests for classification models."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.choice([0, 1, 2], 100)
        return X, y

    def test_random_forest(self, sample_data):
        """Test Random Forest classifier."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10, random_state=42)

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)

        probas = model.predict_proba(X)
        assert probas.shape == (len(y), 3)

    def test_svm(self, sample_data):
        """Test SVM classifier."""
        X, y = sample_data
        model = SVMModel(kernel="rbf", random_state=42)

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_get_model(self):
        """Test model retrieval by name."""
        model = get_model("randomforest", n_estimators=10)
        assert isinstance(model, RandomForestModel)

    def test_list_models(self):
        """Test listing available models."""
        models = list_models()
        assert len(models) > 0
        assert "randomforest" in models or "rf" in models


class TestRegressionModels:
    """Tests for regression models."""

    @pytest.fixture
    def sample_data(self):
        """Create sample regression data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        return X, y

    def test_random_forest_regressor(self, sample_data):
        """Test Random Forest regressor."""
        X, y = sample_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64 or predictions.dtype == np.float32


class TestModelProperties:
    """Tests for model properties."""

    def test_model_params(self):
        """Test getting model parameters."""
        model = RandomForestModel(n_estimators=50, max_depth=10)
        params = model.get_params()

        assert params["name"] == "RandomForest"
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 10

    def test_model_set_params(self):
        """Test setting model parameters."""
        model = RandomForestModel()
        model.set_params(n_estimators=200)

        assert model.params["n_estimators"] == 200

    def test_unfitted_model_prediction(self):
        """Test that unfitted model raises error on predict."""
        model = RandomForestModel()
        X = np.random.randn(10, 50)

        with pytest.raises(ValueError):
            model.predict(X)

