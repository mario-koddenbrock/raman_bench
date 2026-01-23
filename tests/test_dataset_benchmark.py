import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import pytest

# Ensure local package is importable when running tests from repository root
# (insert the 'src' directory at the front of sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _as_dataframe(obj):
    """Coerce loaded object to pandas.DataFrame for test comparisons.

    This handles cases where np.load returns object arrays or 0-d arrays that
    wrap the original DataFrame.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, np.ndarray):
        # 0-d array containing object
        if obj.shape == ():
            return _as_dataframe(obj.item())
        # single-element object array
        if obj.dtype == object and obj.size == 1:
            return _as_dataframe(obj[0])
        # numeric ndarray -> DataFrame
        return pd.DataFrame(obj)
    # Single-element container (e.g., a list) -> unwrap
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return _as_dataframe(obj[0])
    raise TypeError(f"Cannot coerce object of type {type(obj)} to DataFrame")


def test_ramanbenchmark_caching_and_getitem(tmp_path, monkeypatch):
    """Verify RamanBenchmark prepares, caches, and reloads dataset splits.

    The test mocks `raman_data` to return a single synthetic dataset and
    ensures that:
      - `RamanBenchmark` returns pandas DataFrames (or equivalent) for (train, test)
      - Cache files are created in the provided cache directory
      - A second `RamanBenchmark` pointing at the same cache directory loads
        the exact same DataFrames from cache
    """
    # Prepare cache directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Import the module under test so we can monkeypatch attributes on it
    import raman_bench.benchmark.dataset as dataset_mod

    # Mock preprocessing helpers to keep behaviour simple
    monkeypatch.setattr(dataset_mod, "get_preprocessing_pipeline", lambda name: None)
    monkeypatch.setattr(dataset_mod, "handle_missing_values", lambda df: df)

    # Create a deterministic synthetic dataset object
    class SyntheticDataset:
        def __init__(self, n_samples=50, n_features=5):
            rng = np.random.RandomState(0)
            self.targets = rng.rand(n_samples, 1)
            self._X = rng.rand(n_samples, n_features)

        def to_dataframe(self, target_idx: int):
            cols = [f"f{i}" for i in range(self._X.shape[1])]
            df = pd.DataFrame(self._X, columns=cols)
            df["target"] = self.targets[:, target_idx]
            return df

    # Mock raman_data: when called with task_type it should return list of names;
    # when called with a dataset name it should return a SyntheticDataset instance.
    def mock_raman_data(*args, **kwargs):
        if "task_type" in kwargs:
            # return a single dataset name without underscores to avoid split ambiguity
            return ["mockds"]
        if args and isinstance(args[0], str):
            return SyntheticDataset()
        raise ValueError("Unexpected raman_data call")

    monkeypatch.setattr(dataset_mod, "raman_data", mock_raman_data)

    # Now create the benchmark and exercise it
    bench = dataset_mod.RamanBenchmark(n_classification=1, n_regression=0, cache_dir=str(cache_dir), augmentation=False)
    assert len(bench) == 1

    train1, test1 = bench[0]
    train1 = _as_dataframe(train1)
    test1 = _as_dataframe(test1)

    assert isinstance(train1, pd.DataFrame)
    assert isinstance(test1, pd.DataFrame)

    # Ensure cache files were created
    key = bench.key_list[0]
    train_path, test_path = bench.get_cache_file_paths(key)
    assert os.path.exists(train_path), f"Expected cache file {train_path} to exist"
    assert os.path.exists(test_path), f"Expected cache file {test_path} to exist"

    # Create a second benchmark instance pointing to the same cache and verify
    # that the data loaded from cache matches the original DataFrames exactly.
    bench2 = dataset_mod.RamanBenchmark(n_classification=1, n_regression=0, cache_dir=str(cache_dir), augmentation=False)
    train2, test2 = bench2[0]
    train2 = _as_dataframe(train2)
    test2 = _as_dataframe(test2)

    assert_frame_equal(train1, train2)
    assert_frame_equal(test1, test2)
