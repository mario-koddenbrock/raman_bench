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
from raman_data.datasets import pretty_name

logger = logging.getLogger(__name__)


def compute_metrics_from_predictions(config):
    logger.info("\n" + "=" * 60 + "\nSTEP 2: Computing Metrics")
    output_dir = config["output_dir"]
    predictions_dir = os.path.join(output_dir, "predictions")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    benchmark = configure_benchmark(config)

    model_configs = config["models"]
    pbar = tqdm(total=len(benchmark) * len(model_configs))

    classification_output = []
    regression_output = []

    for data_train, data_test, key, task_type in benchmark:
        for model_name in model_configs:
            pbar.set_description(f"{key} | {model_name}")

            filename = f"{key}_{model_name}_predictions.csv"

            if not os.path.exists(os.path.join(predictions_dir, filename)):
                logger.warning(f"Predictions file not found for {key} and model {model_name}. Skipping.")
                continue

            y_pred = pd.read_csv(os.path.join(predictions_dir, filename), index_col=0)

            data_test = data_test.sort_index()
            y_pred = y_pred.sort_index()

            if not np.array_equal(data_test.index, y_pred.index):
                raise ValueError(f"Indices of true values and predictions do not match for {key} and model {model_name}.")

            dataset_name, target_idx, = benchmark.split_key(key)

            output_dict = {
                "key": key,
                "dataset": pretty_name(dataset_name),
                "task_type": task_type,
                "target_idx": target_idx,
                "model": model_name,
            }

            metrics = compute_metrics(data_test["target"], y_pred["target"], task_type=task_type)

            output_dict.update(metrics)
            if task_type == TASK_TYPE.Classification:
                classification_output.append(output_dict)
            else:
                regression_output.append(output_dict)

            pbar.update(1)

    if classification_output:
        classification_df = pd.DataFrame(classification_output)
        classification_df.to_csv(os.path.join(metrics_dir, f"classification_metrics.csv"), index=False)

    if regression_output:
        regression_df = pd.DataFrame(regression_output)
        regression_df.to_csv(os.path.join(metrics_dir, f"regression_metrics.csv"), index=False)




