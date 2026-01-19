"""
Tests for the data module.
"""

import numpy as np
import pytest

from raman_bench.data import DataHandler, RamanDataset


class TestRamanDataset:
    """Tests for the RamanDataset class."""

    def test_dataset_creation(self):
        """Test creating a RamanDataset."""
        data = np.random.randn(100, 500)
        spectra = np.arange(500)
        target = np.random.choice(["A", "B", "C"], 100)

        dataset = RamanDataset(
            name="test",
            data=data,
            spectra=spectra,
            target=target,
            task_type="classification",
        )

        assert dataset.n_samples == 100
        assert dataset.n_features == 500
        assert dataset.n_classes == 3
        assert dataset.task_type == "classification"

    def test_dataset_validation(self):
        """Test dataset validation raises error on mismatch."""
        data = np.random.randn(100, 500)
        spectra = np.arange(500)
        target = np.random.choice(["A", "B"], 50)  # Wrong size

        with pytest.raises(ValueError):
            RamanDataset(
                name="test",
                data=data,
                spectra=spectra,
                target=target,
            )

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        data = np.random.randn(10, 50)
        spectra = np.arange(50)
        target = np.random.choice([0, 1], 10)

        dataset = RamanDataset(
            name="test",
            data=data,
            spectra=spectra,
            target=target,
        )

        df = dataset.to_dataframe()
        assert len(df) == 10
        assert "target" in df.columns
        assert len(df.columns) == 51  # 50 features + 1 target

    def test_train_test_split(self):
        """Test train/test split."""
        data = np.random.randn(100, 50)
        spectra = np.arange(50)
        target = np.random.choice([0, 1], 100)

        dataset = RamanDataset(
            name="test",
            data=data,
            spectra=spectra,
            target=target,
        )

        X_train, X_test, y_train, y_test = dataset.get_train_test_split(
            test_size=0.2, random_state=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20


class TestDataHandler:
    """Tests for the DataHandler class."""

    def test_handler_creation(self):
        """Test creating a DataHandler."""
        handler = DataHandler(cache_datasets=True)
        assert handler.cache_datasets is True

    def test_cache_clearing(self):
        """Test cache clearing."""
        handler = DataHandler(cache_datasets=True)
        handler._cache["test"] = "value"
        handler.clear_cache()
        assert len(handler._cache) == 0

    @pytest.mark.skipif(True, reason="Requires raman_data package")
    def test_list_datasets(self):
        """Test listing datasets."""
        handler = DataHandler()
        datasets = handler.list_datasets()
        assert isinstance(datasets, list)

    @pytest.mark.skipif(True, reason="Requires raman_data package")
    def test_load_dataset(self):
        """Test loading a dataset."""
        handler = DataHandler()
        dataset = handler.load_dataset("Adenine")
        assert isinstance(dataset, RamanDataset)
        assert dataset.n_samples > 0

