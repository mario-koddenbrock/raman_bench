"""
Command-line interface for Raman Bench.
"""

import argparse
import sys
from typing import List, Optional
import logging


logger = logging.getLogger(__name__)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="raman-bench",
        description="Raman Bench: Benchmark evaluation framework for Raman spectroscopy models",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark evaluation")
    run_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )
    run_parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to evaluate",
    )
    run_parser.add_argument(
        "--models",
        nargs="+",
        help="Model names to evaluate",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    run_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate plots from results")
    plot_parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results CSV file or directory",
    )
    plot_parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Output directory for plots",
    )
    plot_parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to plot",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available resources")
    list_parser.add_argument(
        "resource",
        choices=["datasets", "models"],
        help="Resource type to list",
    )
    list_parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        help="Filter by task type",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    # Execute command
    if parsed_args.command == "run":
        return run_benchmark(parsed_args)
    elif parsed_args.command == "plot":
        return generate_plots(parsed_args)
    elif parsed_args.command == "list":
        return list_resources(parsed_args)

    return 0


def run_benchmark(args) -> int:
    """Run benchmark evaluation."""
    from raman_bench.data import DataHandler
    from raman_bench.evaluation import BenchmarkRunner
    from raman_bench.models import get_model, list_models
    from raman_bench.preprocessing import get_default_pipeline

    # Load config if provided
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        datasets = config.get("datasets", [])
        model_configs = config.get("models", [])
    else:
        datasets = args.datasets or []
        model_configs = args.models or ["randomforest", "svm"]

    if not datasets:
        # Use some default datasets
        data_handler = DataHandler()
        datasets = data_handler.list_datasets()[:3]  # First 3 datasets

    # Create models
    models = []
    for model_config in model_configs:
        if isinstance(model_config, str):
            models.append(get_model(model_config))
        elif isinstance(model_config, dict):
            name = model_config.pop("name")
            models.append(get_model(name, **model_config))

    # Create runner
    runner = BenchmarkRunner(
        datasets=datasets,
        models=models,
        preprocessing_pipeline=get_default_pipeline(),
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        results_dir=args.output,
    )

    # Run benchmark
    results = runner.run()

    # Print summary
    logger.info("\n" + runner.summary())

    # Generate plots
    runner.generate_plots()

    return 0


def generate_plots(args) -> int:
    """Generate plots from results."""
    import os
    import pandas as pd
    from raman_bench.plotting import BenchmarkPlotter

    # Load results
    if os.path.isfile(args.results):
        metrics_df = pd.read_csv(args.results)
    else:
        # Look for latest results file
        metrics_file = os.path.join(args.results, "metrics", "benchmark_metrics_latest.csv")
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
        else:
            logger.error(f"Could not find results file at {args.results}")
            return 1

    # Create plotter and generate report
    plotter = BenchmarkPlotter()
    plotter.generate_benchmark_report(
        metrics_df,
        args.output,
        metrics=args.metrics,
    )

    logger.info(f"Plots saved to {args.output}")
    return 0


def list_resources(args) -> int:
    """List available resources."""
    if args.resource == "datasets":
        from raman_bench.data import DataHandler
        handler = DataHandler()
        datasets = handler.list_datasets(task_type=args.task_type)
        logger.info(f"Available datasets ({len(datasets)}):")
        for ds in datasets:
            logger.info(f"  - {ds}")
    elif args.resource == "models":
        from raman_bench.models import list_models
        models = list_models(task_type=args.task_type)
        logger.info(f"Available models ({len(models)}):")
        for model in models:
            logger.info(f"  - {model}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
