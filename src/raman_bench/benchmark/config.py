"""
Configuration and config loading for the benchmark pipeline.
"""
import yaml
import os


DEFAULT_CONFIG = {
    "output_dir": "results",
    "random_state": 42,
    "cv_folds": 5,
    "test_size": 0.2,
    "classification_datasets": ["Adenine"],
    "regression_datasets": [],
    "n_classification_datasets": 0,
    "n_regression_datasets": 0,
    "classification_models": [
        {"name": "RandomForest", "class": "RandomForestClassifier", "tuned": False, "params": {"n_estimators": 100}},
        {"name": "RandomForest (Tuned)", "class": "RandomForestClassifier", "tuned": True, "params": {"n_trials": 30}},
        {"name": "ExtraTrees", "class": "ExtraTreesClassifier", "tuned": False, "params": {"n_estimators": 100}},
        {"name": "ExtraTrees (Tuned)", "class": "ExtraTreesClassifier", "tuned": True, "params": {"n_trials": 30}},
        {"name": "SVM", "class": "SVMClassifier", "tuned": False, "params": {"kernel": "rbf", "C": 1.0}},
        {"name": "SVM (Tuned)", "class": "SVMClassifier", "tuned": True, "params": {"n_trials": 30}},
        {"name": "XGBoost", "class": "XGBoostClassifier", "tuned": False, "params": {"n_estimators": 100}},
        {"name": "XGBoost (Tuned)", "class": "XGBoostClassifier", "tuned": True, "params": {"n_trials": 30}},
        {"name": "LightGBM", "class": "LightGBMClassifier", "tuned": False, "params": {"n_estimators": 100}},
        {"name": "LightGBM (Tuned)", "class": "LightGBMClassifier", "tuned": True, "params": {"n_trials": 30}},
        {"name": "CatBoost", "class": "CatBoostClassifier", "tuned": False, "params": {"iterations": 100, "verbose": False}},
        {"name": "CatBoost (Tuned)", "class": "CatBoostClassifier", "tuned": True, "params": {"verbose": False, "n_trials": 30}},
        {"name": "KNN", "class": "KNNClassifier", "tuned": False, "params": {"n_neighbors": 5}},
        {"name": "KNN (Tuned)", "class": "KNNClassifier", "tuned": True, "params": {"n_trials": 30}},
        {"name": "LogisticRegression", "class": "LinearClassifier", "tuned": False, "params": {"C": 1.0}},
        {"name": "LogisticRegression (Tuned)", "class": "LinearClassifier", "tuned": True, "params": {"n_trials": 30}},
        {"name": "MLP", "class": "MLPClassifier", "tuned": False, "params": {"epochs": 50}},
        {"name": "MLP (Tuned)", "class": "MLPClassifier", "tuned": True, "params": {"epochs": 50, "n_trials": 20}},
        {"name": "TorchMLP", "class": "TorchMLPClassifier", "tuned": False, "params": {"epochs": 50}},
        {"name": "TorchMLP (Tuned)", "class": "TorchMLPClassifier", "tuned": True, "params": {"epochs": 50, "n_trials": 20}},
        {"name": "AutoGluon", "class": "AutoGluonClassifier", "tuned": False, "params": {"time_limit": 300}},
        {"name": "TabPFN_v1", "class": "TabPFNv1Classifier", "tuned": False, "params": {}},
        {"name": "TabPFN_v2", "class": "TabPFNv2Classifier", "tuned": False, "params": {}},
        {"name": "TabPFN_v2.5", "class": "TabPFNv25Classifier", "tuned": False, "params": {}},
        {"name": "EBM", "class": "EBMClassifier", "tuned": False, "params": {}},
        {"name": "EBM (Tuned)", "class": "EBMClassifier", "tuned": True, "params": {"n_trials": 30}},
        {"name": "RealMLP", "class": "RealMLPClassifier", "tuned": False, "params": {}},
        {"name": "TabDPT", "class": "TabDPTClassifier", "tuned": False, "params": {}},
        {"name": "TabM", "class": "TabMClassifier", "tuned": False, "params": {}},
        {"name": "ModernNCA", "class": "ModernNCAClassifier", "tuned": False, "params": {}},
        {"name": "xRFM", "class": "xRFMClassifier", "tuned": False, "params": {}},
        {"name": "SAP-RPT-OSS", "class": "SAPRPTClassifier", "tuned": False, "params": {}},
    ],
    "regression_models": [
        {"name": "RandomForest", "class": "RandomForestRegressor", "tuned": False, "params": {"n_estimators": 100}},
        {"name": "RandomForest (Tuned)", "class": "RandomForestRegressor", "tuned": True, "params": {"n_trials": 30}},
        {"name": "ExtraTrees", "class": "ExtraTreesRegressor", "tuned": False, "params": {"n_estimators": 100}},
        {"name": "SVR", "class": "SVMRegressor", "tuned": False, "params": {"kernel": "rbf", "C": 1.0}},
        {"name": "SVR (Tuned)", "class": "SVMRegressor", "tuned": True, "params": {"n_trials": 30}},
        {"name": "XGBoost", "class": "XGBoostRegressor", "tuned": False, "params": {"n_estimators": 100}},
        {"name": "XGBoost (Tuned)", "class": "XGBoostRegressor", "tuned": True, "params": {"n_trials": 30}},
        {"name": "LightGBM", "class": "LightGBMRegressor", "tuned": False, "params": {"n_estimators": 100}},
        {"name": "LightGBM (Tuned)", "class": "LightGBMRegressor", "tuned": True, "params": {"n_trials": 30}},
        {"name": "CatBoost", "class": "CatBoostRegressor", "tuned": False, "params": {"iterations": 100, "verbose": False}},
        {"name": "KNN", "class": "KNNRegressor", "tuned": False, "params": {"n_neighbors": 5}},
        {"name": "Linear", "class": "LinearRegressor", "tuned": False, "params": {"alpha": 1.0}},
        {"name": "Linear (Tuned)", "class": "LinearRegressor", "tuned": True, "params": {"n_trials": 30}},
        {"name": "MLP", "class": "MLPRegressor", "tuned": False, "params": {"epochs": 50}},
        {"name": "AutoGluon", "class": "AutoGluonRegressor", "tuned": False, "params": {"time_limit": 300}},
        {"name": "EBM", "class": "EBMRegressor", "tuned": False, "params": {}},
    ],
    "preprocessing": "default",
}

def load_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = DEFAULT_CONFIG.copy()
    return config
