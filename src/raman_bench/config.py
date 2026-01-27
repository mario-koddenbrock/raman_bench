"""
Configuration and config loading for the benchmark pipeline.
"""
import yaml
import os


def load_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config
