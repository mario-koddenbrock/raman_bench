"""
Utility and helper functions for the benchmark pipeline.
"""
def check_config(config):
    classification_datasets = config.get("classification_datasets", [])
    classification_models = config.get("classification_models", [])
    regression_datasets = config.get("regression_datasets", [])
    regression_models = config.get("regression_models", [])
    return (len(classification_datasets) * len(classification_models)) + (
        len(regression_datasets) * len(regression_models)
    )
