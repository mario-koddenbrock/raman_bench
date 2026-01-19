"""
Raman Bench: A comprehensive benchmark evaluation framework for Raman spectroscopy models.

This package provides tools for benchmarking machine learning models on Raman spectroscopy data,
including data handling, preprocessing, model implementations, metrics, and visualization.
"""

from raman_bench.evaluation import BenchmarkRunner
from raman_bench.metrics import ClassificationMetrics, RegressionMetrics
from raman_bench.models import get_model, list_models
from raman_bench.plotting import BenchmarkPlotter
from raman_bench.preprocessing import PreprocessingPipeline, get_default_pipeline

__version__ = "0.1.0"
__author__ = "Mario Koddenbrock (HTW Berlin), Christoph Lange (TU Berlin)"

__all__ = [
    "BenchmarkRunner",
    "ClassificationMetrics",
    "RegressionMetrics",
    "get_model",
    "list_models",
    "BenchmarkPlotter",
    "PreprocessingPipeline",
    "get_default_pipeline",
]

