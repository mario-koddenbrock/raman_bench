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
import logging
import os

from raman_bench.config import load_config
from raman_bench.evaluation import compute_metrics_from_predictions
from raman_bench.plotting.plotting import generate_plots_from_metrics
from raman_bench.predictions import compute_predictions

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
        default="configs/minimal.json",
        help="Path to configuration YAML file (default: configs/minimal.json)",
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
    logger.info(f"Running benchmark with configuration: {args.config}")
    config = load_config(args.config)

    config["output_dir"] = args.output
    os.makedirs(config["output_dir"], exist_ok=True)
    logger.info(f"Output directory: {config['output_dir']}")

    if args.step in ["all", "predictions"]:
        compute_predictions(config)

    if args.step in ["all", "metrics"]:
        compute_metrics_from_predictions(config)

    if args.step in ["all", "plots"]:
        generate_plots_from_metrics(config)

    logger.info("\n" + "=" * 60 + "\nBENCHMARK COMPLETE")
    logger.info(f"Results saved to: {config['output_dir']}")

    return 0


if __name__ == "__main__":
    main()
