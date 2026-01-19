#!/usr/bin/env python
"""
Main benchmark evaluation script (refactored).

This script runs the complete benchmark pipeline:
1. Specify models and datasets
2. Compute predictions and save them as CSV
3. Compute metrics based on the CSVs
4. Save metrics in separate CSVs
5. Generate plots based on the metrics CSVs

Usage:
    python run_benchmark.py                    # Run complete benchmark
    python run_benchmark.py --step predictions # Only compute predictions
    python run_benchmark.py --step metrics     # Only compute metrics from predictions
    python run_benchmark.py --step plots       # Only generate plots from metrics
    python run_benchmark.py --config config.yaml  # Use custom configuration
"""

import argparse
import os
import sys
import logging
from raman_bench.benchmark.config import load_config, DEFAULT_CONFIG
from raman_bench.benchmark.predictions import compute_predictions
from raman_bench.benchmark.metrics import compute_metrics_from_predictions
from raman_bench.benchmark.plotting import generate_plots_from_metrics
from raman_bench.benchmark.utils import check_config
from raman_data import raman_data, TASK_TYPE

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    parser = argparse.ArgumentParser(
        description="Run Raman Bench benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--step",
        choices=["all", "predictions", "metrics", "plots"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    config["output_dir"] = args.output
    os.makedirs(config["output_dir"], exist_ok=True)

    n_classification = config.get("n_classification_datasets", 0)
    n_regression = config.get("n_regression_datasets", 0)

    if n_classification == -1:
        config["classification_datasets"] = raman_data(task_type=TASK_TYPE.Classification)
    elif n_classification > 0:
        all_classification = raman_data(task_type=TASK_TYPE.Classification)
        config["classification_datasets"] = all_classification[:n_classification]
    else:
        config["classification_datasets"] = []

    if n_regression == -1:
        config["regression_datasets"] = raman_data(task_type=TASK_TYPE.Regression)
    elif n_regression > 0:
        all_regression = raman_data(task_type=TASK_TYPE.Regression)
        config["regression_datasets"] = all_regression[:n_regression]
    else:
        config["regression_datasets"] = []

    if not check_config(config):
        raise ValueError("No datasets to evaluate. Please check your configuration.")

    if args.step in ["all", "predictions"]:
        compute_predictions(config)

    if args.step in ["all", "metrics"]:
        compute_metrics_from_predictions(config)

    if args.step in ["all", "plots"]:
        generate_plots_from_metrics(config)

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {config['output_dir']}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

