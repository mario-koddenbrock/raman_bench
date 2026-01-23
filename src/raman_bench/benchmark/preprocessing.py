"""
Preprocessing pipeline selection for the benchmark pipeline.
"""
from pandas import DataFrame

from raman_bench.preprocessing import (
    get_default_pipeline, get_minimal_pipeline, get_robust_pipeline
)

def get_preprocessing_pipeline(name: str):
    if name == "default":
        return get_default_pipeline()
    elif name == "minimal":
        return get_minimal_pipeline()
    elif name == "robust":
        return get_robust_pipeline()
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown preprocessing pipeline: {name}")


def handle_missing_values(data: DataFrame, method: str = "drop") -> DataFrame | None:
    # TODO depending on config drop or impute missing values
    if method == "drop":
        return data.dropna()
    elif method == "impute":
        raise NotImplementedError("Imputation not yet implemented.")
    return None

