"""
Main plotter class for benchmark visualization.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class BenchmarkPlotter:
    """
    Plotter for benchmark results visualization.

    Creates publication-ready plots for benchmark analysis.
    """

    def __init__(
        self,
        style: str = "whitegrid",
        context: str = "paper",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        save_format: str = "png",
    ):
        """
        Initialize the benchmark plotter.

        Args:
            style: Seaborn style
            context: Seaborn context
            figsize: Default figure size
            dpi: Resolution for saved figures
            save_format: Default format for saved figures
        """
        self.style = style
        self.context = context
        self.figsize = figsize
        self.dpi = dpi
        self.save_format = save_format

        # Set style
        sns.set_style(style)
        sns.set_context(context)

    def plot_model_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric: str = "accuracy",
        output_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a bar plot comparing models across datasets.

        Args:
            metrics_df: DataFrame with columns ['model', 'dataset', metric]
            metric: Metric to plot
            output_path: Path to save the figure
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create pivot table for grouped bar chart
        if "dataset" in metrics_df.columns:
            # Use pivot_table with aggregation to handle duplicate (dataset, model)
            # entries (e.g., multiple target_idx rows) by averaging the metric.
            try:
                pivot = metrics_df.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
            except Exception:
                # Fall back to grouping and unstacking which provides clearer control
                logger = logging.getLogger(__name__)
                logger.debug("Pivot failed, aggregating by dataset/model using groupby.mean()")
                pivot = metrics_df.groupby(["dataset", "model"]).agg({metric: "mean"}).unstack(level=-1)
            pivot.plot(kind="bar", ax=ax, width=0.8)
            ax.set_xlabel("Dataset")
        else:
            metrics_df.set_index("model")[metric].plot(kind="bar", ax=ax)
            ax.set_xlabel("Model")

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title or f"Model Comparison: {metric.replace('_', ' ').title()}")
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_metrics_heatmap(
        self,
        metrics_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        annot: bool = True,
    ) -> plt.Figure:
        """
        Create a heatmap of metrics across models and datasets.

        Args:
            metrics_df: DataFrame with metrics
            metrics: List of metrics to include (None for all)
            output_path: Path to save the figure
            title: Plot title
            annot: Whether to annotate cells

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Prepare data for heatmap
        if "dataset" in metrics_df.columns:
            # If there are multiple rows for the same (model, dataset)
            # (e.g. different target_idx), aggregate numeric metrics by mean
            # so the heatmap compares models x datasets cleanly.
            group_cols = [c for c in ["model", "dataset"] if c in metrics_df.columns]
            agg_df = metrics_df.groupby(group_cols).mean(numeric_only=True).reset_index()
            agg_df["index"] = agg_df["model"] + " | " + agg_df["dataset"]
            metrics_df = agg_df.set_index("index")

        # Select metrics columns
        if metrics:
            data = metrics_df[metrics]
        else:
            # Select numeric columns
            data = metrics_df.select_dtypes(include=[np.number])

        # Create heatmap
        sns.heatmap(
            data,
            annot=annot,
            fmt=".3f",
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": "Score"},
        )

        ax.set_title(title or "Metrics Heatmap")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_metric_boxplot(
        self,
        metrics_df: pd.DataFrame,
        metric: str = "accuracy",
        group_by: str = "model",
        output_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a boxplot of metrics grouped by model or dataset.

        Args:
            metrics_df: DataFrame with metrics
            metric: Metric to plot
            group_by: Column to group by ('model' or 'dataset')
            output_path: Path to save the figure
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.boxplot(data=metrics_df, x=group_by, y=metric, ax=ax)
        sns.stripplot(
            data=metrics_df, x=group_by, y=metric, ax=ax, color="black", alpha=0.5, size=4
        )

        ax.set_xlabel(group_by.title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title or f"{metric.replace('_', ' ').title()} by {group_by.title()}")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_confusion_matrices(
        self,
        confusion_matrices: Dict[str, np.ndarray],
        class_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        normalize: bool = True,
    ) -> plt.Figure:
        """
        Plot multiple confusion matrices in a grid.

        Args:
            confusion_matrices: Dictionary mapping model names to confusion matrices
            class_names: List of class names
            output_path: Path to save the figure
            normalize: Whether to normalize matrices

        Returns:
            Matplotlib figure
        """
        n_models = len(confusion_matrices)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)

        for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                fmt = ".2f"
            else:
                fmt = "d"

            sns.heatmap(
                cm,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
            )
            ax.set_title(model_name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        # Hide empty subplots
        for idx in range(n_models, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_radar_chart(
        self,
        metrics_df: pd.DataFrame,
        metrics: List[str],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a radar chart comparing models across metrics.

        Args:
            metrics_df: DataFrame with model metrics
            metrics: List of metrics to include
            output_path: Path to save the figure
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(projection="polar"))

        # Number of metrics
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each model
        # If multiple rows per model are present, aggregate by model (mean)
        if "model" in metrics_df.columns:
            radar_df = metrics_df.groupby("model").mean(numeric_only=True)
        else:
            radar_df = metrics_df.set_index("model") if "model" in metrics_df.columns else metrics_df

        for model_name, row in radar_df.iterrows():
            values = row[metrics].tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=str(model_name))
            ax.fill(angles, values, alpha=0.25)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(title or "Model Comparison (Radar Chart)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot training history (loss and metrics over epochs).

        Args:
            history: Dictionary with metric names and their values per epoch
            output_path: Path to save the figure
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for metric_name, values in history.items():
            ax.plot(values, label=metric_name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title(title or "Training History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot predicted vs actual values (for regression).

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            output_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.scatter(y_true, y_pred, alpha=0.5)

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{model_name}: Predicted vs Actual")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot residuals (for regression).

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            output_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color="r", linestyle="--")
        axes[0].set_xlabel("Predicted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title(f"{model_name}: Residual Plot")
        axes[0].grid(True, alpha=0.3)

        # Residual distribution
        axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        axes[1].axvline(x=0, color="r", linestyle="--")
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"{model_name}: Residual Distribution")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def generate_benchmark_report(
        self,
        metrics_df: pd.DataFrame,
        output_dir: str,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Generate a complete set of benchmark plots.

        Args:
            metrics_df: DataFrame with all metrics
            output_dir: Directory to save plots
            metrics: List of metrics to include
        """
        os.makedirs(output_dir, exist_ok=True)

        # Determine metrics
        if metrics is None:
            metrics = [
                col for col in metrics_df.columns
                if col not in ["model", "dataset", "fold"]
            ]

        # Model comparison plots for each metric
        for metric in metrics:
            if metric in metrics_df.columns:
                self.plot_model_comparison(
                    metrics_df,
                    metric=metric,
                    output_path=os.path.join(output_dir, f"comparison_{metric}.{self.save_format}"),
                )

        # Metrics heatmap
        self.plot_metrics_heatmap(
            metrics_df,
            metrics=metrics,
            output_path=os.path.join(output_dir, f"heatmap.{self.save_format}"),
        )

        # Boxplots
        for metric in metrics[:3]:  # Limit to top 3 metrics
            if metric in metrics_df.columns:
                self.plot_metric_boxplot(
                    metrics_df,
                    metric=metric,
                    output_path=os.path.join(output_dir, f"boxplot_{metric}.{self.save_format}"),
                )

        logger = logging.getLogger(__name__)
        logger.info("Benchmark report saved to %s", output_dir)
