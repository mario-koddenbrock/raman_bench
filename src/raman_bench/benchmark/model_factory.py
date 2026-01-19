"""
Model factory and registry for the benchmark pipeline.
"""
from raman_bench.models import (
    RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    SVMClassifier, SVMRegressor, XGBoostClassifier, XGBoostRegressor, LightGBMClassifier, LightGBMRegressor,
    CatBoostClassifier, CatBoostRegressor, KNNClassifier, KNNRegressor, LinearClassifier, LinearRegressor,
    MLPClassifier, MLPRegressor, CNN1DClassifier, CNN1DRegressor, TorchMLPClassifier,
    AutoGluonClassifier, AutoGluonRegressor, TabPFNv1Classifier, TabPFNv2Classifier, TabPFNv25Classifier,
    EBMClassifier, EBMRegressor, RealMLPClassifier, TabDPTClassifier, TabMClassifier, ModernNCAClassifier,
    xRFMClassifier, SAPRPTClassifier
)

MODEL_CLASSES = {
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "SVMClassifier": SVMClassifier,
    "SVMRegressor": SVMRegressor,
    "XGBoostClassifier": XGBoostClassifier,
    "XGBoostRegressor": XGBoostRegressor,
    "LightGBMClassifier": LightGBMClassifier,
    "LightGBMRegressor": LightGBMRegressor,
    "CatBoostClassifier": CatBoostClassifier,
    "CatBoostRegressor": CatBoostRegressor,
    "KNNClassifier": KNNClassifier,
    "KNNRegressor": KNNRegressor,
    "LinearClassifier": LinearClassifier,
    "LinearRegressor": LinearRegressor,
    "MLPClassifier": MLPClassifier,
    "MLPRegressor": MLPRegressor,
    "CNN1DClassifier": CNN1DClassifier,
    "CNN1DRegressor": CNN1DRegressor,
    "TorchMLPClassifier": TorchMLPClassifier,
    "AutoGluonClassifier": AutoGluonClassifier,
    "AutoGluonRegressor": AutoGluonRegressor,
    "TabPFNv1Classifier": TabPFNv1Classifier,
    "TabPFNv2Classifier": TabPFNv2Classifier,
    "TabPFNv25Classifier": TabPFNv25Classifier,
    "EBMClassifier": EBMClassifier,
    "EBMRegressor": EBMRegressor,
    "RealMLPClassifier": RealMLPClassifier,
    "TabDPTClassifier": TabDPTClassifier,
    "TabMClassifier": TabMClassifier,
    "ModernNCAClassifier": ModernNCAClassifier,
    "xRFMClassifier": xRFMClassifier,
    "SAPRPTClassifier": SAPRPTClassifier,
    # Aliases
    "RandomForestModel": RandomForestClassifier,
    "SVMModel": SVMClassifier,
    "SVRModel": SVMRegressor,
}

def create_model(config):
    model_class = MODEL_CLASSES.get(config["class"])
    if model_class is None:
        raise ValueError(f"Unknown model class: {config['class']}")
    tuned = config.get("tuned", False)
    params = config.get("params", {})
    return model_class(tuned=tuned, **params)
