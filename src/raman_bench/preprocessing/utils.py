"""
Utility functions for preprocessing pipelines.

Provides pre-configured pipelines for common use cases.
"""
from raman_bench.preprocessing.pipeline import PreprocessingPipeline
import ramanspy as rp


def get_default_pipeline() -> PreprocessingPipeline:
    """
    Get the default preprocessing pipeline.

    Includes baseline correction, normalization, and smoothing.

    Returns:
        PreprocessingPipeline with default steps
    """
    steps = [
        rp.preprocessing.baseline.ASLS(),
        rp.preprocessing.normalise.MinMax(),
        rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
    ]

    return PreprocessingPipeline(steps=steps, name="default")


def get_minimal_pipeline() -> PreprocessingPipeline:
    """
    Get a minimal preprocessing pipeline.

    Only includes basic normalization.

    Returns:
        PreprocessingPipeline with minimal steps
    """

    steps = [
        rp.preprocessing.normalise.MinMax(),
    ]

    return PreprocessingPipeline(steps=steps, name="minimal")


def get_robust_pipeline() -> PreprocessingPipeline:
    """
    Get a robust preprocessing pipeline.

    Includes cosmic ray removal, baseline correction, normalization, and smoothing.

    Returns:
        PreprocessingPipeline with robust steps
    """

    steps = [
        rp.preprocessing.despike.WhitakerHayes(),
        rp.preprocessing.baseline.ASLS(),
        rp.preprocessing.normalise.Vector(),
        rp.preprocessing.denoise.SavGol(window_length=11, polyorder=3),
    ]

    return PreprocessingPipeline(steps=steps, name="robust")


def get_custom_pipeline(
    baseline: str = "asls",
    normalize: str = "minmax",
    denoise: str = "savgol",
    despike: bool = False,
) -> PreprocessingPipeline:
    """
    Create a custom preprocessing pipeline from named components.

    Args:
        baseline: Baseline correction method ('asls', 'iasls', 'arpl', 'rubberband', 'none')
        normalize: Normalization method ('minmax', 'vector', 'snv', 'none')
        denoise: Denoising method ('savgol', 'gaussian', 'none')
        despike: Whether to include cosmic ray removal

    Returns:
        Custom PreprocessingPipeline
    """

    steps = []

    # Despike
    if despike:
        steps.append(rp.preprocessing.despike.WhitakerHayes())

    # Baseline correction
    baseline_methods = {
        "asls": rp.preprocessing.baseline.ASLS(),
        "iasls": rp.preprocessing.baseline.IASLS(),
        "arpl": rp.preprocessing.baseline.ARPL(),
        "rubberband": rp.preprocessing.baseline.Rubberband(),
        "none": None,
    }
    if baseline.lower() in baseline_methods and baseline_methods[baseline.lower()]:
        steps.append(baseline_methods[baseline.lower()])

    # Normalization
    normalize_methods = {
        "minmax": rp.preprocessing.normalise.MinMax(),
        "vector": rp.preprocessing.normalise.Vector(),
        "snv": rp.preprocessing.normalise.SNV(),
        "none": None,
    }
    if normalize.lower() in normalize_methods and normalize_methods[normalize.lower()]:
        steps.append(normalize_methods[normalize.lower()])

    # Denoising
    denoise_methods = {
        "savgol": rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
        "gaussian": rp.preprocessing.denoise.Gaussian(),
        "none": None,
    }
    if denoise.lower() in denoise_methods and denoise_methods[denoise.lower()]:
        steps.append(denoise_methods[denoise.lower()])

    name = f"custom_{baseline}_{normalize}_{denoise}"
    if despike:
        name = f"custom_despike_{baseline}_{normalize}_{denoise}"

    return PreprocessingPipeline(steps=steps, name=name)

