"""
Plotting module for Raman Bench.

Provides visualization tools for benchmark results.
"""

from raman_bench.plotting.plotter import BenchmarkPlotter
from raman_bench.plotting.utils import (
    plot_confusion_matrix,
    plot_learning_curve,
    plot_metric_comparison,
    plot_roc_curve,
    plot_spectra,
)

__all__ = [
    "BenchmarkPlotter",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_metric_comparison",
    "plot_spectra",
    "plot_learning_curve",
]

