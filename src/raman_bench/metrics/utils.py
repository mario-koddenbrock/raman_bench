"""
Utility functions for metrics computation.
"""

from typing import Any, Dict, Optional

import numpy as np

from raman_bench.metrics.classification import ClassificationMetrics
from raman_bench.metrics.regression import RegressionMetrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "classification",
    y_proba: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Compute metrics based on task type.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: Type of task ('classification' or 'regression')
        y_proba: Predicted probabilities (for classification)
        **kwargs: Additional arguments for metric calculators

    Returns:
        Dictionary of metric names to values
    """
    if task_type.lower() == "classification":
        calculator = ClassificationMetrics(**kwargs)
        return calculator.compute_all(y_true, y_pred, y_proba)
    elif task_type.lower() == "regression":
        calculator = RegressionMetrics()
        return calculator.compute_all(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def compare_metrics(
    results: Dict[str, Dict[str, float]],
    metric: str,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Compare metrics across multiple models/experiments.

    Args:
        results: Dictionary mapping model names to their metrics
        metric: Name of the metric to compare
        higher_is_better: Whether higher values are better

    Returns:
        Dictionary with comparison results
    """
    values = {}
    for model_name, metrics in results.items():
        if metric in metrics:
            values[model_name] = metrics[metric]

    if not values:
        return {"error": f"Metric '{metric}' not found in any results"}

    sorted_models = sorted(values.keys(), key=lambda x: values[x], reverse=higher_is_better)

    return {
        "best_model": sorted_models[0],
        "best_value": values[sorted_models[0]],
        "worst_model": sorted_models[-1],
        "worst_value": values[sorted_models[-1]],
        "ranking": sorted_models,
        "values": values,
        "mean": np.mean(list(values.values())),
        "std": np.std(list(values.values())),
    }


def get_metric_info(metric_name: str) -> Dict[str, Any]:
    """
    Get information about a metric.

    Args:
        metric_name: Name of the metric

    Returns:
        Dictionary with metric information
    """
    metric_info = {
        # Classification metrics
        "accuracy": {
            "name": "Accuracy",
            "higher_is_better": True,
            "range": (0, 1),
            "description": "Proportion of correctly classified samples",
        },
        "precision": {
            "name": "Precision",
            "higher_is_better": True,
            "range": (0, 1),
            "description": "Proportion of true positives among predicted positives",
        },
        "recall": {
            "name": "Recall",
            "higher_is_better": True,
            "range": (0, 1),
            "description": "Proportion of true positives among actual positives",
        },
        "f1_score": {
            "name": "F1 Score",
            "higher_is_better": True,
            "range": (0, 1),
            "description": "Harmonic mean of precision and recall",
        },
        "balanced_accuracy": {
            "name": "Balanced Accuracy",
            "higher_is_better": True,
            "range": (0, 1),
            "description": "Average of recall for each class",
        },
        "roc_auc": {
            "name": "ROC-AUC",
            "higher_is_better": True,
            "range": (0, 1),
            "description": "Area under the ROC curve",
        },
        "cohen_kappa": {
            "name": "Cohen's Kappa",
            "higher_is_better": True,
            "range": (-1, 1),
            "description": "Agreement between predictions and true labels, adjusted for chance",
        },
        "matthews_corrcoef": {
            "name": "Matthews Correlation Coefficient",
            "higher_is_better": True,
            "range": (-1, 1),
            "description": "Correlation between predictions and true labels",
        },
        # Regression metrics
        "mse": {
            "name": "Mean Squared Error",
            "higher_is_better": False,
            "range": (0, float("inf")),
            "description": "Average of squared differences",
        },
        "rmse": {
            "name": "Root Mean Squared Error",
            "higher_is_better": False,
            "range": (0, float("inf")),
            "description": "Square root of MSE",
        },
        "mae": {
            "name": "Mean Absolute Error",
            "higher_is_better": False,
            "range": (0, float("inf")),
            "description": "Average of absolute differences",
        },
        "r2": {
            "name": "R² Score",
            "higher_is_better": True,
            "range": (float("-inf"), 1),
            "description": "Proportion of variance explained",
        },
        "mape": {
            "name": "Mean Absolute Percentage Error",
            "higher_is_better": False,
            "range": (0, float("inf")),
            "description": "Average of absolute percentage differences",
        },
        "explained_variance": {
            "name": "Explained Variance",
            "higher_is_better": True,
            "range": (float("-inf"), 1),
            "description": "Proportion of variance explained",
        },
        "max_error": {
            "name": "Maximum Error",
            "higher_is_better": False,
            "range": (0, float("inf")),
            "description": "Maximum absolute difference",
        },
        "median_ae": {
            "name": "Median Absolute Error",
            "higher_is_better": False,
            "range": (0, float("inf")),
            "description": "Median of absolute differences",
        },
    }

    return metric_info.get(
        metric_name,
        {"name": metric_name, "higher_is_better": True, "range": None, "description": "Unknown metric"},
    )

