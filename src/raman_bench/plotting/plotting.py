"""
Plot generation and summary reporting for the benchmark pipeline.
"""
import os
import logging
import pandas as pd
from datetime import datetime
from raman_bench.plotting import BenchmarkPlotter

logger = logging.getLogger(__name__)

def generate_plots_from_metrics(config):
    logger.info("\n" + "=" * 60 + "\nSTEP 3: Generating Plots")
    output_dir = config["output_dir"]
    metrics_dir = os.path.join(output_dir, "metrics")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, "benchmark_metrics_latest.csv")
    if not os.path.exists(metrics_file):
        logger.error(f"Metrics file not found: {metrics_file}")
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    elif not os.path.getsize(metrics_file):
        logger.error(f"Metrics file is empty: {metrics_file}")
        raise ValueError(f"Metrics file is empty: {metrics_file}")
    metrics_df = pd.read_csv(metrics_file)
    plotter = BenchmarkPlotter(dpi=150, save_format="png")
    classification_metrics = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
    classification_df = metrics_df[metrics_df["task_type"] == TASK_TYPE.Classification]
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
    regression_df = metrics_df[metrics_df["task_type"] == "regression"]
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
    _generate_summary_report(metrics_df, plots_dir)
    logger.info(f"\nPlots saved to: {plots_dir}")

def _generate_summary_report(metrics_df, plots_dir):
    report_lines = [
        "=" * 60,
        "BENCHMARK SUMMARY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"Total evaluations: {len(metrics_df)}",
        f"Datasets: {metrics_df['dataset'].nunique()}",
        f"Models: {metrics_df['model'].nunique()}",
        "",
    ]
    clf_df = metrics_df[metrics_df["task_type"] == "classification"]
    if not clf_df.empty:
        report_lines.extend([
            "CLASSIFICATION RESULTS",
            "-" * 40,
        ])
        for model in clf_df["model"].unique():
            model_df = clf_df[clf_df["model"] == model]
            acc_mean = model_df["accuracy"].mean()
            acc_std = model_df["accuracy"].std() if len(model_df) > 1 else 0
            f1_mean = model_df.get("f1_score", pd.Series([0])).mean()
            report_lines.append(
                f"  {model}: Acc={acc_mean:.4f}±{acc_std:.4f}, F1={f1_mean:.4f}"
            )
        report_lines.append("")
    reg_df = metrics_df[metrics_df["task_type"] == "regression"]
    if not reg_df.empty:
        report_lines.extend([
            "REGRESSION RESULTS",
            "-" * 40,
        ])
        for model in reg_df["model"].unique():
            model_df = reg_df[reg_df["model"] == model]
            r2_mean = model_df["r2"].mean()
            r2_std = model_df["r2"].std() if len(model_df) > 1 else 0
            rmse_mean = model_df.get("rmse", pd.Series([0])).mean()
            report_lines.append(
                f"  {model}: R²={r2_mean:.4f}±{r2_std:.4f}, RMSE={rmse_mean:.4f}"
            )
        report_lines.append("")
    report_lines.extend([
        "BEST MODELS",
        "-" * 40,
    ])
    if not clf_df.empty:
        best_clf = clf_df.loc[clf_df["accuracy"].idxmax()]
        report_lines.append(
            f"  Classification: {best_clf['model']} on {best_clf['dataset']} "
            f"(Acc={best_clf['accuracy']:.4f})"
        )
    if not reg_df.empty:
        best_reg = reg_df.loc[reg_df["r2"].idxmax()]
        report_lines.append(
            f"  Regression: {best_reg['model']} on {best_reg['dataset']} "
            f"(R²={best_reg['r2']:.4f})"
        )
    report_path = os.path.join(plots_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    logger.info("\n" + "\n".join(report_lines))
