"""
Plot generation and summary reporting for the benchmark pipeline.
"""
import logging
import os

import pandas as pd

from raman_bench.plotting import BenchmarkPlotter

logger = logging.getLogger(__name__)

def generate_plots_from_metrics(config):
    logger.info("\n" + "=" * 60 + "\nSTEP 3: Generating Plots")
    output_dir = config["output_dir"]

    metrics_dir = os.path.join(output_dir, "metrics")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metrics_file_classification = os.path.join(metrics_dir, "classification_metrics.csv")
    metrics_file_regression = os.path.join(metrics_dir, "regression_metrics.csv")

    if not os.path.exists(metrics_file_classification):
        logger.error(f"Metrics file not found: {metrics_file_classification}")
        raise FileNotFoundError(f"Metrics file not found: {metrics_file_classification}")
    elif not os.path.getsize(metrics_file_classification):
        logger.error(f"Metrics file is empty: {metrics_file_classification}")
        raise ValueError(f"Metrics file is empty: {metrics_file_classification}")

    if not os.path.exists(metrics_file_regression):
        logger.error(f"Metrics file not found: {metrics_file_regression}")
        raise FileNotFoundError(f"Metrics file not found: {metrics_file_regression}")
    elif not os.path.getsize(metrics_file_regression):
        logger.error(f"Metrics file is empty: {metrics_file_regression}")
        raise ValueError(f"Metrics file is empty: {metrics_file_regression}")

    classification_df = pd.read_csv(metrics_file_classification)
    regression_df = pd.read_csv(metrics_file_regression)

    plotter = BenchmarkPlotter(dpi=150, save_format="png")
    classification_metrics = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]

    if not classification_df.empty:
        logger.info("\nGenerating classification plots...")
        clf_plots_dir = os.path.join(plots_dir, "classification")
        os.makedirs(clf_plots_dir, exist_ok=True)
        for metric in classification_metrics:
            if metric in classification_df.columns:
                plotter.plot_model_comparison(
                    classification_df,
                    metric=metric,
                    output_path=os.path.join(clf_plots_dir, f"comparison_{metric}.png"),
                    title=f"Model Comparison: {metric.replace('_', ' ').title()}",
                )
        available_metrics = [m for m in classification_metrics if m in classification_df.columns]
        plotter.plot_metrics_heatmap(
            classification_df,
            metrics=available_metrics,
            output_path=os.path.join(clf_plots_dir, "heatmap.png"),
            title="Classification Metrics Heatmap",
        )
        for metric in ["accuracy", "f1_score"]:
            if metric in classification_df.columns:
                plotter.plot_metric_boxplot(
                    classification_df,
                    metric=metric,
                    output_path=os.path.join(clf_plots_dir, f"boxplot_{metric}.png"),
                )
    regression_metrics = ["r2", "rmse", "mae", "mape"]
    if not regression_df.empty:
        logger.info("\nGenerating regression plots...")
        reg_plots_dir = os.path.join(plots_dir, "regression")
        os.makedirs(reg_plots_dir, exist_ok=True)
        for metric in regression_metrics:
            if metric in regression_df.columns:
                plotter.plot_model_comparison(
                    regression_df,
                    metric=metric,
                    output_path=os.path.join(reg_plots_dir, f"comparison_{metric}.png"),
                    title=f"Model Comparison: {metric.upper()}",
                )
        available_metrics = [m for m in regression_metrics if m in regression_df.columns]
        plotter.plot_metrics_heatmap(
            regression_df,
            metrics=available_metrics,
            output_path=os.path.join(reg_plots_dir, "heatmap.png"),
            title="Regression Metrics Heatmap",
        )

    logger.info(f"\nPlots saved to: {plots_dir}")
