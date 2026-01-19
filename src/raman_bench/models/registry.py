"""
Model registry for Raman Bench.

Provides functions to register, retrieve, and list available models.
"""

from typing import Any, Dict, List, Optional, Type

from raman_bench.models.base import BaseModel

# Global model registry
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """
    Register a model class in the registry.

    Args:
        name: Name to register the model under
        model_class: Model class to register
    """
    _MODEL_REGISTRY[name.lower()] = model_class


def get_model(name: str, tuned: bool = False, **kwargs) -> BaseModel:
    """
    Get a model instance by name.

    Args:
        name: Name of the model
        tuned: Whether to use HPO-tuned version
        **kwargs: Parameters to pass to the model constructor

    Returns:
        Model instance

    Raises:
        ValueError: If model name is not found in registry
    """
    name_lower = name.lower()

    # Check for "_tuned" suffix
    if name_lower.endswith("_tuned"):
        name_lower = name_lower[:-6]
        tuned = True

    if name_lower not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found. Available models: {list_models()}"
        )

    return _MODEL_REGISTRY[name_lower](tuned=tuned, **kwargs)


def list_models(task_type: Optional[str] = None, include_tuned: bool = False) -> List[str]:
    """
    List all registered models.

    Args:
        task_type: Filter by task type ('classification' or 'regression')
        include_tuned: Whether to include tuned variants in the list

    Returns:
        List of model names
    """
    models = []

    for name, model_class in _MODEL_REGISTRY.items():
        try:
            model = model_class()
            if task_type is None or model.task_type == task_type.lower():
                models.append(name)
                if include_tuned and model_class.supports_tuning:
                    models.append(f"{name}_tuned")
        except Exception:
            # If we can't instantiate, include it anyway
            models.append(name)

    return sorted(set(models))


def get_all_models(task_type: str = "classification", include_tuned: bool = True) -> List[BaseModel]:
    """
    Get instances of all registered models for a task type.

    Args:
        task_type: Task type ('classification' or 'regression')
        include_tuned: Whether to include tuned variants

    Returns:
        List of model instances
    """
    models = []
    seen_names = set()

    for name, model_class in _MODEL_REGISTRY.items():
        try:
            model = model_class(tuned=False)
            if model.task_type == task_type.lower():
                # Default version
                if model.name not in seen_names:
                    models.append(model)
                    seen_names.add(model.name)

                    # Tuned version if supported
                    if include_tuned and model_class.supports_tuning:
                        tuned_model = model_class(tuned=True)
                        models.append(tuned_model)
        except Exception:
            pass

    return models


def _register_default_models():
    """Register all default models."""
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

    # Register classification models
    register_model("randomforest", RandomForestClassifier)
    register_model("rf", RandomForestClassifier)
    register_model("extratrees", ExtraTreesClassifier)
    register_model("et", ExtraTreesClassifier)
    register_model("svm", SVMClassifier)
    register_model("xgboost", XGBoostClassifier)
    register_model("xgb", XGBoostClassifier)
    register_model("lightgbm", LightGBMClassifier)
    register_model("lgbm", LightGBMClassifier)
    register_model("catboost", CatBoostClassifier)
    register_model("cb", CatBoostClassifier)
    register_model("knn", KNNClassifier)
    register_model("logistic", LinearClassifier)
    register_model("logistic_regression", LinearClassifier)
    register_model("linear", LinearClassifier)
    register_model("mlp", MLPClassifier)
    register_model("cnn1d", CNN1DClassifier)
    register_model("cnn", CNN1DClassifier)

    # Advanced classification models
    register_model("autogluon", AutoGluonClassifier)
    register_model("ag", AutoGluonClassifier)
    register_model("tabpfn", TabPFNv1Classifier)
    register_model("tabpfn_v1", TabPFNv1Classifier)
    register_model("tabpfnv1", TabPFNv1Classifier)
    register_model("tabpfn_v2", TabPFNv2Classifier)
    register_model("tabpfnv2", TabPFNv2Classifier)
    register_model("tabpfn_v2.5", TabPFNv25Classifier)
    register_model("tabpfnv25", TabPFNv25Classifier)
    register_model("ebm", EBMClassifier)
    register_model("realmlp", RealMLPClassifier)
    register_model("tabdpt", TabDPTClassifier)
    register_model("tabm", TabMClassifier)
    register_model("modernnca", ModernNCAClassifier)
    register_model("nca", ModernNCAClassifier)
    register_model("xrfm", xRFMClassifier)
    register_model("torchmlp", TorchMLPClassifier)
    register_model("saprpt", SAPRPTClassifier)
    register_model("sap_rpt_oss", SAPRPTClassifier)

    # Register regression models
    register_model("randomforest_regressor", RandomForestRegressor)
    register_model("rf_regressor", RandomForestRegressor)
    register_model("extratrees_regressor", ExtraTreesRegressor)
    register_model("et_regressor", ExtraTreesRegressor)
    register_model("svr", SVMRegressor)
    register_model("svm_regressor", SVMRegressor)
    register_model("xgboost_regressor", XGBoostRegressor)
    register_model("xgb_regressor", XGBoostRegressor)
    register_model("lightgbm_regressor", LightGBMRegressor)
    register_model("lgbm_regressor", LightGBMRegressor)
    register_model("catboost_regressor", CatBoostRegressor)
    register_model("cb_regressor", CatBoostRegressor)
    register_model("knn_regressor", KNNRegressor)
    register_model("linear_regressor", LinearRegressor)
    register_model("ridge", LinearRegressor)
    register_model("mlp_regressor", MLPRegressor)
    register_model("cnn1d_regressor", CNN1DRegressor)
    register_model("cnn_regressor", CNN1DRegressor)

    # Advanced regression models
    register_model("autogluon_regressor", AutoGluonRegressor)
    register_model("ag_regressor", AutoGluonRegressor)
    register_model("tabpfn_v2_regressor", TabPFNv2Regressor)
    register_model("ebm_regressor", EBMRegressor)


# Register default models on import
_register_default_models()

