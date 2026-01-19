"""
Tests for the preprocessing module.
"""

import numpy as np
import pytest

from raman_bench.preprocessing import (
    PreprocessingPipeline,
    get_default_pipeline,
    get_minimal_pipeline,
    get_robust_pipeline,
)
from raman_bench.data import RamanDataset


class TestPreprocessingPipeline:
    """Tests for the PreprocessingPipeline class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectral data."""
        np.random.seed(42)
        X = np.abs(np.random.randn(10, 100)) + 1
        wavenumbers = np.linspace(400, 2000, 100)
        return X, wavenumbers

    def test_identity_pipeline(self, sample_data):
        """Test identity pipeline returns data unchanged."""
        from raman_bench.preprocessing.pipeline import IdentityPipeline

        X, wavenumbers = sample_data
        pipeline = IdentityPipeline()

        X_transformed = pipeline.fit_transform(X, wavenumbers=wavenumbers)

        np.testing.assert_array_equal(X, X_transformed)

    def test_empty_pipeline(self, sample_data):
        """Test empty pipeline returns data unchanged."""
        X, wavenumbers = sample_data
        pipeline = PreprocessingPipeline(steps=[], name="empty")

        X_transformed = pipeline.fit_transform(X, wavenumbers=wavenumbers)

        np.testing.assert_array_equal(X, X_transformed)

    @pytest.mark.skipif(True, reason="Requires ramanspy package")
    def test_default_pipeline(self, sample_data):
        """Test default preprocessing pipeline."""
        X, wavenumbers = sample_data
        pipeline = get_default_pipeline()

        X_transformed = pipeline.fit_transform(X, wavenumbers=wavenumbers)

        assert X_transformed.shape == X.shape
        assert not np.array_equal(X, X_transformed)

    @pytest.mark.skipif(True, reason="Requires ramanspy package")
    def test_minimal_pipeline(self, sample_data):
        """Test minimal preprocessing pipeline."""
        X, wavenumbers = sample_data
        pipeline = get_minimal_pipeline()

        X_transformed = pipeline.fit_transform(X, wavenumbers=wavenumbers)

        assert X_transformed.shape == X.shape

    def test_pipeline_params(self):
        """Test getting pipeline parameters."""
        pipeline = PreprocessingPipeline(steps=[], name="test")
        params = pipeline.get_params()

        assert params["name"] == "test"
        assert params["n_steps"] == 0

    def test_add_step(self):
        """Test adding steps to pipeline."""
        pipeline = PreprocessingPipeline(steps=[], name="test")
        assert len(pipeline.steps) == 0

        pipeline.add_step("dummy_step")
        assert len(pipeline.steps) == 1


class TestDatasetTransform:
    """Tests for transforming datasets."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        np.random.seed(42)
        data = np.abs(np.random.randn(10, 100)) + 1
        spectra = np.linspace(400, 2000, 100)
        target = np.random.choice([0, 1], 10)

        return RamanDataset(
            name="test",
            data=data,
            spectra=spectra,
            target=target,
        )

    def test_transform_dataset_preserves_target(self, sample_dataset):
        """Test that transforming dataset preserves target."""
        from raman_bench.preprocessing.pipeline import IdentityPipeline

        pipeline = IdentityPipeline()
        transformed = pipeline.transform_dataset(sample_dataset)

        np.testing.assert_array_equal(
            sample_dataset.targets,
            transformed.targets,
        )

    def test_transform_dataset_preserves_metadata(self, sample_dataset):
        """Test that transforming dataset updates metadata."""
        from raman_bench.preprocessing.pipeline import IdentityPipeline

        pipeline = IdentityPipeline()
        transformed = pipeline.transform_dataset(sample_dataset)

        assert transformed.metadata.get("preprocessed") is True
        assert transformed.metadata.get("pipeline") == "identity"

