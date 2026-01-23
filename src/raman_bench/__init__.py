"""
Raman Bench: A comprehensive benchmark evaluation framework for Raman spectroscopy models.

This package provides tools for benchmarking machine learning models on Raman spectroscopy data,
including data handling, preprocessing, model implementations, metrics, and visualization.
"""

from raman_bench.metrics import ClassificationMetrics, RegressionMetrics
from raman_bench.plotting import BenchmarkPlotter
from raman_bench.preprocessing import PreprocessingPipeline, get_default_pipeline

__version__ = "0.1.0"
__author__ = "Mario Koddenbrock (HTW Berlin), Christoph Lange (TU Berlin)"

__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "BenchmarkPlotter",
    "PreprocessingPipeline",
    "get_default_pipeline",
]

