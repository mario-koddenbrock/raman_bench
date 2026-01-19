"""
Metrics computation for the benchmark pipeline.
"""
import os
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
from raman_bench.metrics import compute_metrics

logger = logging.getLogger(__name__)


def compute_metrics_from_predictions(config):
    logger.info("\n" + "=" * 60 + "\nSTEP 2: Computing Metrics")
    output_dir = config["output_dir"]
    predictions_dir = os.path.join(output_dir, "predictions")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    if not os.path.exists(predictions_dir):
        logger.error(f"Predictions directory not found: {predictions_dir}")
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    elif not os.listdir(predictions_dir):
        logger.error(f"No prediction files found in directory: {predictions_dir}")
        raise ValueError(f"No prediction files found in directory: {predictions_dir}")

    all_metrics = []
    prediction_files = list(Path(predictions_dir).glob("*_predictions.csv"))
    logger.info(f"Found {len(prediction_files)} prediction files")
    for pred_file in prediction_files:
        pred_df = pd.read_csv(pred_file)
        y_true = pred_df["y_true"].values
        y_pred = pred_df["y_pred"].values
        proba_cols = [c for c in pred_df.columns if c.startswith("proba")]
        y_proba = pred_df[proba_cols].values if proba_cols else None
        task_type = _infer_task_type(y_true)
        filename = pred_file.stem.replace("_predictions", "")
        parts = filename.rsplit("_", 1)
        if len(parts) == 2:
            dataset_name, model_name = parts
        else:
            dataset_name, model_name = filename, "unknown"
        metrics = compute_metrics(y_true, y_pred, task_type=task_type, y_proba=y_proba)
        metrics["dataset"] = dataset_name
        metrics["model"] = model_name
        metrics["task_type"] = task_type
        metrics["n_samples"] = len(y_true)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_df.to_csv(os.path.join(metrics_dir, f"benchmark_metrics_{timestamp}.csv"), index=False)
    metrics_df.to_csv(os.path.join(metrics_dir, "benchmark_metrics_latest.csv"), index=False)

    if len(metrics_df) > 0:
        for task_type in metrics_df["task_type"].unique():
            task_df = metrics_df[metrics_df["task_type"] == task_type]
            task_df.to_csv(os.path.join(metrics_dir, f"metrics_{task_type}.csv"), index=False)
        logger.info(f"\nMetrics saved to: {metrics_dir}")
    else:
        logger.warning("\nNo metrics computed. Skipping saving metrics.")

    return metrics_df


def _infer_task_type(y):
    if hasattr(y, 'dtype') and y.dtype.kind in ["U", "S", "O"]:
        return TASK_TYPE.Classification
    unique_values = np.unique(y)
    if len(unique_values) < 20:
        return TASK_TYPE.Classification
    return TASK_TYPE.Regression
