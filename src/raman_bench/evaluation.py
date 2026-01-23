"""
Metrics computation for the benchmark pipeline.
"""
import os
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm

from raman_bench.benchmark import configure_benchmark
from raman_bench.metrics import compute_metrics
from raman_data import TASK_TYPE

logger = logging.getLogger(__name__)


def compute_metrics_from_predictions(config):
    logger.info("\n" + "=" * 60 + "\nSTEP 2: Computing Metrics")
    output_dir = config["output_dir"]
    predictions_dir = os.path.join(output_dir, "predictions")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    benchmark = configure_benchmark(config)

    model_configs = config["model_configs"]
    pbar = tqdm(total=len(benchmark) * len(model_configs))

    for data_train, data_test, key, task_type in benchmark:
        for model_config in model_configs:
            model_name = model_config["name"]
            pbar.set_description(f"{key} | {model_name}")

            filename = f"{key}_{model_name}_predictions.csv"
            y_pred = pd.read_csv(os.path.join(predictions_dir, filename))["y_pred"]

            metrics = compute_metrics(data_test["y_true"], y_pred, task_type=task_type)
            metrics["dataset"] = key
            metrics["model"] = model_name
            metrics["task_type"] = task_type
            metrics["n_samples"] = len(data_test["y_true"])
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(os.path.join(metrics_dir, f"metrics_{key}_{model_name}.csv"), index=False)
            pbar.update(1)




