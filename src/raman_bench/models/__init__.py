"""
Models module for Raman Bench.

Provides machine learning models for classification and regression tasks.
Each model supports both default and tuned (HPO) versions.
"""

from raman_bench.models.base import BaseModel, ClassificationModel, RegressionModel

# Classical models
from raman_bench.models.classical import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    SVMClassifier,
    SVMRegressor,
    XGBoostClassifier,
    XGBoostRegressor,
    LightGBMClassifier,
    LightGBMRegressor,
    CatBoostClassifier,
    CatBoostRegressor,
    KNNClassifier,
    KNNRegressor,
    LinearClassifier,
    LinearRegressor,
    # Aliases for backward compatibility
    RandomForestModel,
    SVMModel,
    SVRModel,
)

# Deep learning models
from raman_bench.models.deep_learning import (
    MLPClassifier,
    MLPRegressor,
    CNN1DClassifier,
    CNN1DRegressor,
)

# Advanced models
from raman_bench.models.advanced import (
    AutoGluonClassifier,
    AutoGluonRegressor,
    TabPFNv1Classifier,
    TabPFNv2Classifier,
    TabPFNv2Regressor,
    TabPFNv25Classifier,
    EBMClassifier,
    EBMRegressor,
    RealMLPClassifier,
    TabDPTClassifier,
    TabMClassifier,
    ModernNCAClassifier,
    xRFMClassifier,
    TorchMLPClassifier,
    SAPRPTClassifier,
)

# Registry functions
from raman_bench.models.registry import (
    get_model,
    list_models,
    register_model,
    get_all_models,
)

__all__ = [
    # Base classes
    "BaseModel",
    "ClassificationModel",
    "RegressionModel",
    # Classical models
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "SVMClassifier",
    "SVMRegressor",
    "XGBoostClassifier",
    "XGBoostRegressor",
    "LightGBMClassifier",
    "LightGBMRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "KNNClassifier",
    "KNNRegressor",
    "LinearClassifier",
    "LinearRegressor",
    # Aliases
    "RandomForestModel",
    "SVMModel",
    "SVRModel",
    # Deep learning models
    "MLPClassifier",
    "MLPRegressor",
    "CNN1DClassifier",
    "CNN1DRegressor",
    # Advanced models
    "AutoGluonClassifier",
    "AutoGluonRegressor",
    "TabPFNv1Classifier",
    "TabPFNv2Classifier",
    "TabPFNv2Regressor",
    "TabPFNv25Classifier",
    "EBMClassifier",
    "EBMRegressor",
    "RealMLPClassifier",
    "TabDPTClassifier",
    "TabMClassifier",
    "ModernNCAClassifier",
    "xRFMClassifier",
    "TorchMLPClassifier",
    "SAPRPTClassifier",
    # Registry functions
    "get_model",
    "list_models",
    "register_model",
    "get_all_models",
]

