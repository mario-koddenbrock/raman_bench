"""
Benchmark runner for evaluating models on Raman spectroscopy datasets.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from raman_bench.metrics import compute_metrics
from raman_bench.models.base import BaseModel
from raman_bench.plotting import BenchmarkPlotter
from raman_bench.preprocessing import PreprocessingPipeline, get_default_pipeline
from raman_data import RamanDataset


class BenchmarkRunner:
    """
    Runner for executing benchmark evaluations.

    Coordinates data loading, preprocessing, model training, evaluation, and result storage.
    """

    def __init__(
        self,
        datasets: Optional[List[str]] = None,
        models: Optional[List[BaseModel]] = None,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        results_dir: str = "results",
        verbose: bool = True,
    ):
        """
        Initialize the benchmark runner.

        Args:
            datasets: List of dataset names to evaluate on
            models: List of model instances to evaluate
            preprocessing_pipeline: Preprocessing pipeline to apply
            cv_folds: Number of cross-validation folds
            test_size: Test set size for train/test split
            random_state: Random seed for reproducibility
            results_dir: Directory to save results
            verbose: Whether to print progress
        """
        self.datasets = datasets or []
        self.models = models or []
        self.preprocessing_pipeline = preprocessing_pipeline or get_default_pipeline()
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.results_dir = results_dir
        self.verbose = verbose

        # Initialize components
        self.plotter = BenchmarkPlotter()

        # Results storage
        self.results: Dict[str, Dict] = {}
        self.predictions: Dict[str, Dict] = {}
        self.metrics_df: Optional[pd.DataFrame] = None

    def add_dataset(self, dataset_name: str) -> "BenchmarkRunner":
        """Add a dataset to the benchmark."""
        if dataset_name not in self.datasets:
            self.datasets.append(dataset_name)
        return self

    def add_model(self, model: BaseModel) -> "BenchmarkRunner":
        """Add a model to the benchmark."""
        self.models.append(model)
        return self

    def set_preprocessing(self, pipeline: PreprocessingPipeline) -> "BenchmarkRunner":
        """Set the preprocessing pipeline."""
        self.preprocessing_pipeline = pipeline
        return self

    def run(
        self,
        use_cv: bool = True,
        save_predictions: bool = True,
    ) -> pd.DataFrame:
        """
        Run the benchmark evaluation.

        Args:
            use_cv: Whether to use cross-validation
            save_predictions: Whether to save predictions

        Returns:
            DataFrame with metrics for all model-dataset combinations
        """
        if not self.datasets:
            raise ValueError("No datasets specified. Add datasets with add_dataset()")
        if not self.models:
            raise ValueError("No models specified. Add models with add_model()")

        all_results = []

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Progress bar
        total_iterations = len(self.datasets) * len(self.models)
        pbar = tqdm(total=total_iterations, disable=not self.verbose)

        for dataset_name in self.datasets:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Dataset: {dataset_name}")
                print(f"{'='*60}")

            # Load dataset
            try:
                dataset = self.data_handler.load_dataset(dataset_name)
            except Exception as e:
                print(f"Failed to load dataset {dataset_name}: {e}")
                continue

            # Apply preprocessing
            if self.preprocessing_pipeline:
                dataset = self.preprocessing_pipeline.transform_dataset(dataset)

            for model in self.models:
                pbar.set_description(f"{dataset_name} | {model.name}")

                try:
                    if use_cv:
                        result = self._evaluate_cv(dataset, model)
                    else:
                        result = self._evaluate_holdout(dataset, model)

                    result["dataset"] = dataset_name
                    result["model"] = model.name
                    result["task_type"] = dataset.task_type
                    result["n_samples"] = dataset.n_samples
                    result["n_features"] = dataset.n_features

                    all_results.append(result)

                    # Store predictions
                    if save_predictions and "predictions" in result:
                        self._save_predictions(
                            dataset_name,
                            model.name,
                            result["predictions"],
                        )

                except Exception as e:
                    print(f"Error evaluating {model.name} on {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()

                pbar.update(1)

        pbar.close()

        # Compile results
        self.metrics_df = pd.DataFrame(all_results)
        self._save_metrics()

        return self.metrics_df

    def _evaluate_cv(
        self,
        dataset: RamanDataset,
        model: BaseModel,
    ) -> Dict[str, Any]:
        """
        Evaluate model using cross-validation.

        Args:
            dataset: Dataset to evaluate on
            model: Model to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        from copy import deepcopy

        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []

        # Select CV strategy
        if dataset.task_type == "classification":
            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            split_generator = cv.split(dataset.data, dataset.targets)
        else:
            cv = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            split_generator = cv.split(dataset.data)

        for fold, (train_idx, test_idx) in enumerate(split_generator):
            X_train, X_test = dataset.data[train_idx], dataset.data[test_idx]
            y_train, y_test = dataset.targets[train_idx], dataset.targets[test_idx]

            # Create fresh model instance
            model_copy = deepcopy(model)

            # Train
            start_time = time.time()
            model_copy.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Predict
            start_time = time.time()
            y_pred = model_copy.predict(X_test)
            predict_time = time.time() - start_time

            # Get probabilities if available
            y_proba = model_copy.predict_proba(X_test)

            # Compute metrics
            metrics = compute_metrics(
                y_test, y_pred,
                task_type=dataset.task_type,
                y_proba=y_proba,
            )
            metrics["train_time"] = train_time
            metrics["predict_time"] = predict_time
            metrics["fold"] = fold

            fold_metrics.append(metrics)

            # Collect predictions
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            if y_proba is not None:
                all_y_proba.extend(y_proba)

        # Aggregate metrics
        result = self._aggregate_metrics(fold_metrics)

        # Store predictions
        result["predictions"] = {
            "y_true": np.array(all_y_true),
            "y_pred": np.array(all_y_pred),
            "y_proba": np.array(all_y_proba) if all_y_proba else None,
        }

        return result

    def _evaluate_holdout(
        self,
        dataset: RamanDataset,
        model: BaseModel,
    ) -> Dict[str, Any]:
        """
        Evaluate model using holdout validation.

        Args:
            dataset: Dataset to evaluate on
            model: Model to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        from copy import deepcopy

        # Split data
        X_train, X_test, y_train, y_test = dataset.get_train_test_split(
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Create fresh model instance
        model_copy = deepcopy(model)

        # Train
        start_time = time.time()
        model_copy.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict
        start_time = time.time()
        y_pred = model_copy.predict(X_test)
        predict_time = time.time() - start_time

        # Get probabilities if available
        y_proba = model_copy.predict_proba(X_test)

        # Compute metrics
        result = compute_metrics(
            y_test, y_pred,
            task_type=dataset.task_type,
            y_proba=y_proba,
        )
        result["train_time"] = train_time
        result["predict_time"] = predict_time

        # Store predictions
        result["predictions"] = {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        return result

    def _aggregate_metrics(
        self,
        fold_metrics: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Aggregate metrics across folds.

        Args:
            fold_metrics: List of metric dictionaries from each fold

        Returns:
            Aggregated metrics with mean and std
        """
        result = {}

        # Get all metric names
        metric_names = set()
        for fm in fold_metrics:
            metric_names.update(fm.keys())

        for metric in metric_names:
            if metric == "fold":
                continue

            values = [fm.get(metric, np.nan) for fm in fold_metrics]
            values = [v for v in values if not np.isnan(v)]

            if values:
                result[metric] = np.mean(values)
                result[f"{metric}_std"] = np.std(values)

        return result

    def _save_predictions(
        self,
        dataset_name: str,
        model_name: str,
        predictions: Dict[str, np.ndarray],
    ) -> None:
        """Save predictions to CSV."""
        predictions_dir = os.path.join(self.results_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        # Create safe filename
        safe_dataset = dataset_name.replace("/", "_").replace("\\", "_")
        filename = f"{safe_dataset}_{model_name}_predictions.csv"

        df = pd.DataFrame({
            "y_true": predictions["y_true"],
            "y_pred": predictions["y_pred"],
        })

        if predictions.get("y_proba") is not None:
            proba = predictions["y_proba"]
            if proba.ndim == 2:
                for i in range(proba.shape[1]):
                    df[f"proba_class_{i}"] = proba[:, i]
            else:
                df["proba"] = proba

        df.to_csv(os.path.join(predictions_dir, filename), index=False)

    def _save_metrics(self) -> None:
        """Save metrics to CSV."""
        if self.metrics_df is None or self.metrics_df.empty:
            return

        metrics_dir = os.path.join(self.results_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_metrics_{timestamp}.csv"

        self.metrics_df.to_csv(
            os.path.join(metrics_dir, filename),
            index=False,
        )

        # Also save latest version
        self.metrics_df.to_csv(
            os.path.join(metrics_dir, "benchmark_metrics_latest.csv"),
            index=False,
        )

    def load_results(self, filepath: str) -> pd.DataFrame:
        """
        Load results from a CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            DataFrame with loaded results
        """
        self.metrics_df = pd.read_csv(filepath)
        return self.metrics_df

    def generate_plots(
        self,
        output_dir: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Generate benchmark plots.

        Args:
            output_dir: Directory to save plots (defaults to results_dir/plots)
            metrics: List of metrics to plot
        """
        if self.metrics_df is None or self.metrics_df.empty:
            raise ValueError("No results available. Run benchmark first or load results.")

        output_dir = output_dir or os.path.join(self.results_dir, "plots")
        self.plotter.generate_benchmark_report(
            self.metrics_df,
            output_dir,
            metrics=metrics,
        )

    def get_best_models(
        self,
        metric: str = "accuracy",
        per_dataset: bool = True,
    ) -> pd.DataFrame:
        """
        Get the best models based on a metric.

        Args:
            metric: Metric to rank by
            per_dataset: Whether to get best model per dataset

        Returns:
            DataFrame with best models
        """
        if self.metrics_df is None or self.metrics_df.empty:
            raise ValueError("No results available. Run benchmark first.")

        if metric not in self.metrics_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        if per_dataset:
            idx = self.metrics_df.groupby("dataset")[metric].idxmax()
            return self.metrics_df.loc[idx]
        else:
            idx = self.metrics_df[metric].idxmax()
            return self.metrics_df.loc[[idx]]

    def summary(self) -> str:
        """
        Generate a text summary of benchmark results.

        Returns:
            Summary string
        """
        if self.metrics_df is None or self.metrics_df.empty:
            return "No results available."

        lines = [
            "=" * 60,
            "BENCHMARK SUMMARY",
            "=" * 60,
            f"Datasets: {len(self.metrics_df['dataset'].unique())}",
            f"Models: {len(self.metrics_df['model'].unique())}",
            f"Total evaluations: {len(self.metrics_df)}",
            "",
        ]

        # Classification metrics
        if "accuracy" in self.metrics_df.columns:
            lines.append("Classification Results (mean ± std):")
            lines.append("-" * 40)
            for model in self.metrics_df["model"].unique():
                model_data = self.metrics_df[self.metrics_df["model"] == model]
                acc = model_data["accuracy"].mean()
                acc_std = model_data["accuracy"].std()
                lines.append(f"  {model}: {acc:.4f} ± {acc_std:.4f}")

        # Regression metrics
        if "r2" in self.metrics_df.columns:
            lines.append("\nRegression Results (mean ± std):")
            lines.append("-" * 40)
            for model in self.metrics_df["model"].unique():
                model_data = self.metrics_df[self.metrics_df["model"] == model]
                r2 = model_data["r2"].mean()
                r2_std = model_data["r2"].std()
                lines.append(f"  {model}: R² = {r2:.4f} ± {r2_std:.4f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BenchmarkRunner(datasets={len(self.datasets)}, "
            f"models={len(self.models)}, cv_folds={self.cv_folds})"
        )

