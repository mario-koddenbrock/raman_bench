"""
Raman Bench: A comprehensive benchmark evaluation framework for Raman spectroscopy models.

This package provides tools for benchmarking machine learning models on Raman spectroscopy data,
including data handling, preprocessing, model implementations, metrics, and visualization.

Top-level names are lazily imported on attribute access to avoid importing heavy optional
dependencies (for example AutoGluon) during a simple `import raman_bench`.
"""

import importlib
from typing import List

__version__ = "0.1.0"
__author__ = "Mario Koddenbrock (HTW Berlin), Christoph Lange (TU Berlin)"

# Public API names and the module path they are provided from. Values are tuples of
# (module_name, attribute_name).
_public_map = {
    "ClassificationMetrics": ("raman_bench.metrics", "ClassificationMetrics"),
    "RegressionMetrics": ("raman_bench.metrics", "RegressionMetrics"),
    "compute_metrics": ("raman_bench.metrics", "compute_metrics"),
    "BenchmarkPlotter": ("raman_bench.plotting", "BenchmarkPlotter"),
    "RamanBenchmark": ("raman_bench.benchmark", "RamanBenchmark"),
    "AutoGluonModel": ("raman_bench.model", "AutoGluonModel"),
    "load_config": ("raman_bench.config", "load_config"),
    "DEFAULT_CONFIG": ("raman_bench.config", "DEFAULT_CONFIG"),
    "compute_predictions": ("raman_bench.predictions", "compute_predictions"),
    "compute_metrics_from_predictions": ("raman_bench.evaluation", "compute_metrics_from_predictions"),
    "get_preprocessing_pipeline": ("raman_bench.preprocessing", "get_preprocessing_pipeline"),
}

__all__: List[str] = sorted(list(_public_map.keys()))


def __getattr__(name: str):
    """Lazily import and return a public attribute.

    This avoids importing potentially heavy optional dependencies at package import
    time and keeps the top-level import lightweight.
    """
    if name in _public_map:
        module_name, attr = _public_map[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_public_map.keys()))
