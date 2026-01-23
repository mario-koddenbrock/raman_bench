"""Dataset loading and caching utilities for Raman Bench.

This module exposes the `RamanBenchmark` class which loads datasets from the
`raman_data` package, applies preprocessing and optional augmentation, splits
into train/test sets, and caches prepared dataset splits to disk for faster
subsequent runs. The class is intentionally lightweight and focused on the
benchmark workflow used by the runner.
"""

import json
import logging
import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from raman_bench.preprocessing import get_preprocessing_pipeline
from raman_data import raman_data, TASK_TYPE

logger = logging.getLogger(__name__)


class RamanBenchmark:
    """Manage dataset loading, preprocessing, caching and iteration for the benchmark.

    The `RamanBenchmark` class provides an iterator-like container of prepared
    train/test splits for datasets sourced from the `raman_data` package. It
    supports both classification and regression datasets and caches prepared
    numpy files on disk to avoid repeated preprocessing work.

    Key responsibilities:
      - Query available datasets using `raman_data()` and select a subset for
        classification and regression benchmarks
      - Load the raw dataset, optionally apply a preprocessing pipeline and
        augmentation, convert to pandas DataFrame, and split into train/test
      - Persist splits to a simple on-disk cache (numpy .npy files) and maintain
        a JSON index describing how many target splits exist per dataset
      - Provide sequence-style access via `__len__` and `__getitem__` so the
        benchmark runner can iterate all prepared splits

    Parameters
    ----------
    n_classification : int, optional
        Number of classification datasets to load. Use -1 to load all available
        classification datasets. Defaults to 1.
    n_regression : int, optional
        Number of regression datasets to load. Use -1 to load all available
        regression datasets. Defaults to 1.
    test_size : float, optional
        Fraction of the dataset to reserve for the test split (passed to
        sklearn.model_selection.train_test_split). Defaults to 0.2.
    random_state : int, optional
        Random seed used for the train/test split. Defaults to 42.
    preprocessing : str, optional
        Name of the preprocessing pipeline to use. This is passed to
        `get_preprocessing_pipeline`. Defaults to "default".
    augmentation : bool, optional
        Whether to perform data augmentation. Currently a placeholder flag; if
        True a message is logged and the dataset is returned unchanged.
    cache_dir : str, optional
        Directory used to store cached .npy dataset splits and the index.json
        file. Defaults to ".cache".

    Attributes
    ----------
    dataset_names_classification : list[str]
        Names of classification datasets chosen for the benchmark.
    dataset_names_regression : list[str]
        Names of regression datasets chosen for the benchmark.
    key_list : list[str]
        Flat list of keys identifying each dataset split in the format
        "{dataset_name}_{target_idx}". This is the sequence order used by
        `__getitem__`.
    index : dict
        Mapping from dataset name to number of target columns (i.e. how many
        target splits exist for that dataset). Persisted in `index.json`.

    Examples
    --------
    >>> bench = RamanBenchmark(n_classification=2, n_regression=0, cache_dir=".cache")
    >>> len(bench)
    2
    >>> train_df, test_df = bench[0]

    """

    def __init__(
            self,
            n_classification: int = 1,
            n_regression: int = 1,
            test_size: float = 0.2,
            random_state: int = 42,
            preprocessing: bool = False,
            augmentation: bool = False,
            cache_dir: str = ".cache",
    ):

        self.n_classification = n_classification
        self.n_regression = n_regression
        logger.info(f"Datasets: {n_classification} classification, {n_regression} regression")

        self.test_size = test_size
        self.random_state = random_state

        self.dataset_names_classification = []
        if n_classification == -1:
            self.dataset_names_classification = raman_data(task_type=TASK_TYPE.Classification)
        elif n_classification > 0:
            all_classification = raman_data(task_type=TASK_TYPE.Classification)
            # self.dataset_names_classification = all_classification[:n_classification]
            self.dataset_names_classification = [all_classification[-1]] # TODO

        self.dataset_names_regression = []
        if n_regression == -1:
            self.dataset_names_regression = raman_data(task_type=TASK_TYPE.Regression)
        elif n_regression > 0:
            all_regression = raman_data(task_type=TASK_TYPE.Regression)
            self.dataset_names_regression = all_regression[:n_regression]

        self.preprocessing = preprocessing
        self.preprocessing_pipeline = get_preprocessing_pipeline()

        self.augmentation = augmentation
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.key_list = []
        self.task_type_list = []

        self.index_file = os.path.join(cache_dir, "index.json")
        self.index:Dict[str, int] = self.load_index()
        self.init_datasets()
        self.save_index()



    def __len__(self):
        """Return number of prepared dataset splits available in the benchmark.

        This corresponds to the length of ``self.key_list``.
        """
        return len(self.key_list)

    def __getitem__(self, idx: int) -> Tuple[DataFrame, DataFrame, str, TASK_TYPE]:
        """Return the (train, test) DataFrame pair for the given index.

        The index corresponds to entries in ``self.key_list``. If the dataset
        split has been cached on disk it is loaded from the cache; otherwise it
        will be prepared on-the-fly, saved to cache and returned.

        Parameters
        ----------
        idx : int
            Integer index into the list of prepared splits.

        Returns
        -------
        (pandas.DataFrame, pandas.DataFrame)
            Tuple containing (train_dataframe, test_dataframe).
        """
        key = self.key_list[idx]
        task_type = self.task_type_list[idx]
        if self.has_dataset_in_cache(key):
            data_train, data_test = self.load_dataset_from_cache(key)
        else:
            # this should not happen
            data_train, data_test = self.load_dataset_from_key(key)
            self.save_dataset(key, data_train, data_test)
            logger.warning(f"Dataset {key} already in cache.")
        return data_train, data_test, key, task_type


    @staticmethod
    def get_key(dataset_name: str, target_idx: int) -> str:
        """Create a key string for a dataset target split.

        The returned format is ``"{dataset_name}_{target_idx}"`` and is used to
        identify and cache specific target columns from multi-target datasets.
        """
        dataset_name = dataset_name.replace("/", "_").replace("\\", "_")
        return f"{dataset_name}_{target_idx}"

    @staticmethod
    def split_key(key: str) -> Tuple[str, int]:
        """Reverse `get_key` and return (dataset_name, target_idx).

        Raises
        ------
        ValueError
            If the key cannot be split into the expected two parts using
            underscore delimiting.
        """
        splits = key.split("_")
        dataset_name = "_".join(splits[:-1])
        target_idx = splits[-1]
        return dataset_name, int(target_idx)

    def load_index(self) -> Dict[str, int]:
        """Load the JSON index file from disk if present.

        The index maps dataset names to their number of target columns. If the
        index file does not exist an empty dict is returned.
        """
        if not os.path.exists(self.index_file):
            return dict()
        with open(self.index_file, "r") as f:
            index = json.load(f)
        return index

    def save_index(self):
        """Persist the in-memory index mapping to disk (``index.json``)."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f)

    def init_datasets(self):
        """Ensure dataset cache and key list are initialized.

        This will scan the configured classification and regression dataset
        names, populate the index with target counts for each dataset, and
        construct ``self.key_list`` for iteration.
        """
        self.load_datasets(self.dataset_names_classification)
        self.load_datasets(self.dataset_names_regression)

        for dataset_name in self.dataset_names_classification + self.dataset_names_regression:
            for target_idx in range(self.index[dataset_name]):
                self.key_list.append(self.get_key(dataset_name, target_idx))
                self.task_type_list.append(TASK_TYPE.Classification if dataset_name in self.dataset_names_classification else TASK_TYPE.Regression)


    def get_cache_file_paths(self, key: str) -> Tuple[str, str]:
        """Return the file paths for the train/test .npy cache files for a key.

        Returns
        -------
        (train_path, test_path)
        """
        return f"{self.cache_dir}/{key}_train.pkl", f"{self.cache_dir}/{key}_test.pkl"

    def save_dataset(self, key: str, data_train: DataFrame, data_test: DataFrame):
        """Save train/test split to disk as numpy files using allow_pickle=True.

        Note: DataFrames are saved as pickled numpy objects; when loading they
        will need to be interpreted back into DataFrames by callers.
        """
        cache_file_train, cache_file_test = self.get_cache_file_paths(key)
        data_train.to_pickle(cache_file_train)
        data_test.to_pickle(cache_file_test)

    def load_dataset_from_cache(self, key: str) -> Tuple[DataFrame, DataFrame]:
        """Load a previously cached train/test split from disk and return it.

        Returns
        -------
        (train, test) : tuple
            The loaded numpy objects (typically pandas DataFrames) for train and
            test splits.
        """
        cache_file_train, cache_file_test = self.get_cache_file_paths(key)
        data_train = pd.read_pickle(cache_file_train)
        data_test = pd.read_pickle(cache_file_test)
        return data_train, data_test

    def has_dataset_in_cache(self, key: str) -> bool:
        """Return True if both train and test cache files exist for key."""
        cache_file_train, cache_file_test = self.get_cache_file_paths(key)
        return os.path.exists(cache_file_train) and os.path.exists(cache_file_test)

    def load_dataset_from_key(self, key: str) -> Tuple[DataFrame, DataFrame]:
        """Load and prepare a dataset split identified by the given key.

        This performs the following steps:
          - Resolve dataset name and target index via ``split_key``
          - Load the dataset object via ``raman_data(dataset_name)``
          - Optionally apply augmentation (currently a placeholder)
          - Apply the configured preprocessing pipeline
          - Convert to a pandas DataFrame for the requested target index
          - Handle missing values (via ``handle_missing_values``)
          - Split into train/test with ``train_test_split``

        Returns
        -------
        (train_df, test_df)
        """
        dataset_name, target_idx = self.split_key(key)
        dataset = raman_data(dataset_name)

        # TODO Data augmentation
        if self.augmentation:
            print("Data augmentation not yet implemented.")

        if self.preprocessing:
            dataset = self.preprocessing_pipeline(dataset)

        num_targets = dataset.targets.shape[1] if dataset.targets.ndim > 1 else 1

        if target_idx >= num_targets:
            raise ValueError(f"Target index {target_idx} is out of range for dataset {dataset_name}")

        data_df = dataset.to_dataframe(target_idx)
        data_df = self.handle_missing_values(data_df)

        data_train, data_test = train_test_split(
            data_df,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        return data_train, data_test

    def load_datasets(self, dataset_names:List[str]):

        for dataset_name in tqdm(dataset_names, desc=f"Loading datasets"):
            if dataset_name in self.index:
                num_targets = self.index[dataset_name]
            else:
                dataset = raman_data(dataset_name)
                num_targets = dataset.targets.shape[1] if dataset.targets.ndim > 1 else 1
                self.index[dataset_name] = num_targets

            for target_idx in range(num_targets):
                key = self.get_key(dataset_name, target_idx)
                if self.has_dataset_in_cache(key):
                    logger.debug(f"Dataset {key} already in cache.")
                else:
                    data_train, data_test = self.load_dataset_from_key(key)
                    self.save_dataset(key, data_train, data_test)

    def handle_missing_values(self, data_df: DataFrame) -> DataFrame:
        data_df = data_df.dropna()
        # TODO imputation?
        return data_df

