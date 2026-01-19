"""
Classical machine learning models for Raman spectroscopy.

Includes Random Forest, SVM, XGBoost, LightGBM, CatBoost, and other classical ML models.
Each model supports both default and tuned (HPO) versions.
"""

from typing import Any, Dict, Optional

import numpy as np

from raman_bench.models.base import ClassificationModel, RegressionModel


# =============================================================================
# Random Forest
# =============================================================================

class RandomForestClassifier(ClassificationModel):
    """Random Forest classifier for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="RandomForest",
            tuned=tuned,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return SKRandomForestClassifier(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class RandomForestRegressor(RegressionModel):
    """Random Forest regressor for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="RandomForest",
            tuned=tuned,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return SKRandomForestRegressor(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


# =============================================================================
# Extra Trees
# =============================================================================

class ExtraTreesClassifier(ClassificationModel):
    """Extra Trees classifier for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="ExtraTrees",
            tuned=tuned,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.ensemble import ExtraTreesClassifier as SKExtraTreesClassifier
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return SKExtraTreesClassifier(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesClassifier":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class ExtraTreesRegressor(RegressionModel):
    """Extra Trees regressor for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="ExtraTrees",
            tuned=tuned,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.ensemble import ExtraTreesRegressor as SKExtraTreesRegressor
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return SKExtraTreesRegressor(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesRegressor":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


# =============================================================================
# SVM / SVR
# =============================================================================

class SVMClassifier(ClassificationModel):
    """Support Vector Machine classifier for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "C": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]},
    }

    def __init__(
        self,
        tuned: bool = False,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        probability: bool = True,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(
            name="SVM",
            tuned=tuned,
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.svm import SVC
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return SVC(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.params.get("probability", True):
            return self.model.predict_proba(X)
        return None


class SVMRegressor(RegressionModel):
    """Support Vector Regressor for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "C": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]},
        "epsilon": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        epsilon: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            name="SVR",
            tuned=tuned,
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.svm import SVR
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return SVR(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMRegressor":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


# =============================================================================
# XGBoost
# =============================================================================

class XGBoostClassifier(ClassificationModel):
    """XGBoost classifier for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="XGBoost",
            tuned=tuned,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from xgboost import XGBClassifier
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return XGBClassifier(**merged_params, use_label_encoder=False, eval_metric="mlogloss")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class XGBoostRegressor(RegressionModel):
    """XGBoost regressor for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="XGBoost",
            tuned=tuned,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from xgboost import XGBRegressor
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return XGBRegressor(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


# =============================================================================
# LightGBM
# =============================================================================

class LightGBMClassifier(ClassificationModel):
    """LightGBM classifier for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "num_leaves": {"type": "int", "low": 10, "high": 100},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        verbose: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="LightGBM",
            tuned=tuned,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )

    def _create_model(self, **params):
        from lightgbm import LGBMClassifier
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return LGBMClassifier(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMClassifier":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class LightGBMRegressor(RegressionModel):
    """LightGBM regressor for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "num_leaves": {"type": "int", "low": 10, "high": 100},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        verbose: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="LightGBM",
            tuned=tuned,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )

    def _create_model(self, **params):
        from lightgbm import LGBMRegressor
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return LGBMRegressor(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMRegressor":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


# =============================================================================
# CatBoost
# =============================================================================

class CatBoostClassifier(ClassificationModel):
    """CatBoost classifier for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "iterations": {"type": "int", "low": 50, "high": 500},
        "depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "l2_leaf_reg": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0},
    }

    def __init__(
        self,
        tuned: bool = False,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        random_state: Optional[int] = 42,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            name="CatBoost",
            tuned=tuned,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )

    def _create_model(self, **params):
        from catboost import CatBoostClassifier as CBClassifier
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return CBClassifier(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostClassifier":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class CatBoostRegressor(RegressionModel):
    """CatBoost regressor for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "iterations": {"type": "int", "low": 50, "high": 500},
        "depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "l2_leaf_reg": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        random_state: Optional[int] = 42,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            name="CatBoost",
            tuned=tuned,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )

    def _create_model(self, **params):
        from catboost import CatBoostRegressor as CBRegressor
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return CBRegressor(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostRegressor":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


# =============================================================================
# KNN
# =============================================================================

class KNNClassifier(ClassificationModel):
    """K-Nearest Neighbors classifier for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_neighbors": {"type": "int", "low": 1, "high": 50},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
        "metric": {"type": "categorical", "choices": ["euclidean", "manhattan", "minkowski"]},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "minkowski",
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="KNN",
            tuned=tuned,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.neighbors import KNeighborsClassifier
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        merged_params.pop("random_state", None)
        return KNeighborsClassifier(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class KNNRegressor(RegressionModel):
    """K-Nearest Neighbors regressor for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_neighbors": {"type": "int", "low": 1, "high": 50},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
        "metric": {"type": "categorical", "choices": ["euclidean", "manhattan", "minkowski"]},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "minkowski",
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="KNN",
            tuned=tuned,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.neighbors import KNeighborsRegressor
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        merged_params.pop("random_state", None)
        return KNeighborsRegressor(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


# =============================================================================
# Linear Models
# =============================================================================

class LinearClassifier(ClassificationModel):
    """Logistic Regression classifier for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "C": {"type": "float", "low": 1e-4, "high": 1e4, "log": True},
        "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
    }

    def __init__(
        self,
        tuned: bool = False,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="LogisticRegression",
            tuned=tuned,
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.linear_model import LogisticRegression
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        # Adjust solver based on penalty
        if merged_params.get("penalty") == "l1":
            merged_params["solver"] = "saga"
        return LogisticRegression(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearClassifier":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class LinearRegressor(RegressionModel):
    """Linear Regression (Ridge) for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "alpha": {"type": "float", "low": 1e-4, "high": 1e4, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        alpha: float = 1.0,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(
            name="Linear",
            tuned=tuned,
            alpha=alpha,
            random_state=random_state,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.linear_model import Ridge
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return Ridge(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            self.model = self._create_model(**self._best_params)
        else:
            self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


# Aliases for backward compatibility
RandomForestModel = RandomForestClassifier
SVMModel = SVMClassifier
SVRModel = SVMRegressor

