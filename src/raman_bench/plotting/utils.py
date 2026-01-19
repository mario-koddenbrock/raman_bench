"""
Utility functions for plotting.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_spectra(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Raman Spectra",
    xlabel: str = "Raman Shift (cm⁻¹)",
    ylabel: str = "Intensity (a.u.)",
    figsize: Tuple[int, int] = (12, 6),
    alpha: float = 0.5,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Raman spectra.

    Args:
        spectra: Spectral data (n_samples, n_features)
        wavenumbers: Wavenumber values
        labels: Class labels for coloring (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        alpha: Line transparency
        output_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            for spectrum in spectra[mask]:
                ax.plot(wavenumbers, spectrum, color=color, alpha=alpha, label=label)

        # Create legend with unique labels
        handles, legend_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(legend_labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Class")
    else:
        for spectrum in spectra:
            ax.plot(wavenumbers, spectrum, alpha=alpha)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        output_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        output_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import LabelBinarizer

    fig, ax = plt.subplots(figsize=figsize)

    n_classes = len(np.unique(y_true))

    if n_classes == 2:
        # Binary classification
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    else:
        # Multi-class
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)

        if class_names is None:
            class_names = [str(c) for c in lb.classes_]

        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metric_comparison(
    results: dict,
    metric: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a bar plot comparing a metric across models.

    Args:
        results: Dictionary mapping model names to metrics dictionaries
        metric: Metric to plot
        title: Plot title
        figsize: Figure size
        output_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    bars = ax.bar(models, values)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Model Comparison: {metric.replace('_', ' ').title()}")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    title: str = "Learning Curve",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot learning curve.

    Args:
        train_sizes: Training set sizes
        train_scores: Training scores
        test_scores: Test/validation scores
        title: Plot title
        figsize: Figure size
        output_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="blue",
    )
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2,
        color="orange",
    )
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
    ax.plot(train_sizes, test_mean, "o-", color="orange", label="Cross-validation score")

    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig

