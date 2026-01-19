"""
Advanced and tabular deep learning models for Raman spectroscopy.

Includes AutoGluon, TabPFN variants, EBM, and other advanced tabular models.
"""

from typing import Any, Dict, Optional

import numpy as np

from raman_bench.models.base import ClassificationModel, RegressionModel


# =============================================================================
# AutoGluon
# =============================================================================

class AutoGluonClassifier(ClassificationModel):
    """AutoGluon TabularPredictor for classification."""

    # AutoGluon does its own HPO internally, so no external tuning needed
    supports_tuning = False

    def __init__(
        self,
        tuned: bool = False,  # Ignored - AutoGluon always does HPO
        time_limit: int = 300,
        presets: str = "medium_quality",
        random_state: Optional[int] = 42,
        verbosity: int = 0,
        **kwargs,
    ):
        super().__init__(
            name="AutoGluon",
            tuned=False,  # AutoGluon handles its own tuning
            time_limit=time_limit,
            presets=presets,
            random_state=random_state,
            verbosity=verbosity,
            **kwargs,
        )
        self._temp_dir = None

    def _create_model(self, **params):
        # AutoGluon model is created during fit
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoGluonClassifier":
        import tempfile
        import pandas as pd
        from autogluon.tabular import TabularPredictor

        # Create DataFrame
        df = pd.DataFrame(X)
        df["target"] = y

        # Create temp directory for AutoGluon
        self._temp_dir = tempfile.mkdtemp()

        self.model = TabularPredictor(
            label="target",
            path=self._temp_dir,
            verbosity=self.params.get("verbosity", 0),
        )
        self.model.fit(
            df,
            time_limit=self.params.get("time_limit", 300),
            presets=self.params.get("presets", "medium_quality"),
        )
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        df = pd.DataFrame(X)
        return self.model.predict(df).values

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        df = pd.DataFrame(X)
        return self.model.predict_proba(df).values


class AutoGluonRegressor(RegressionModel):
    """AutoGluon TabularPredictor for regression."""

    supports_tuning = False

    def __init__(
        self,
        tuned: bool = False,
        time_limit: int = 300,
        presets: str = "medium_quality",
        random_state: Optional[int] = 42,
        verbosity: int = 0,
        **kwargs,
    ):
        super().__init__(
            name="AutoGluon",
            tuned=False,
            time_limit=time_limit,
            presets=presets,
            random_state=random_state,
            verbosity=verbosity,
            **kwargs,
        )
        self._temp_dir = None

    def _create_model(self, **params):
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoGluonRegressor":
        import tempfile
        import pandas as pd
        from autogluon.tabular import TabularPredictor

        df = pd.DataFrame(X)
        df["target"] = y

        self._temp_dir = tempfile.mkdtemp()

        self.model = TabularPredictor(
            label="target",
            path=self._temp_dir,
            problem_type="regression",
            verbosity=self.params.get("verbosity", 0),
        )
        self.model.fit(
            df,
            time_limit=self.params.get("time_limit", 300),
            presets=self.params.get("presets", "medium_quality"),
        )
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        df = pd.DataFrame(X)
        return self.model.predict(df).values


# =============================================================================
# TabPFN v1
# =============================================================================

class TabPFNv1Classifier(ClassificationModel):
    """TabPFN v1 classifier - no training required, uses pretrained transformer."""

    # TabPFN has no parameters to tune
    supports_tuning = False

    def __init__(
        self,
        tuned: bool = False,
        N_ensemble_configurations: int = 32,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            name="TabPFN_v1",
            tuned=False,
            N_ensemble_configurations=N_ensemble_configurations,
            device=device,
            **kwargs,
        )

    def _create_model(self, **params):
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(
            N_ensemble_configurations=self.params.get("N_ensemble_configurations", 32),
            device=self.params.get("device", "cpu"),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNv1Classifier":
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


# =============================================================================
# TabPFN v2 (placeholder - actual implementation depends on package availability)
# =============================================================================

class TabPFNv2Classifier(ClassificationModel):
    """TabPFN v2 classifier with improved architecture."""

    supports_tuning = False

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 8,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            name="TabPFN_v2",
            tuned=False,
            n_estimators=n_estimators,
            device=device,
            **kwargs,
        )

    def _create_model(self, **params):
        try:
            from tabpfn_v2 import TabPFNClassifier as TabPFNv2
            return TabPFNv2(
                n_estimators=self.params.get("n_estimators", 8),
                device=self.params.get("device", "cpu"),
            )
        except ImportError:
            # Fallback to v1 if v2 not available
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier(
                N_ensemble_configurations=self.params.get("n_estimators", 8) * 4,
                device=self.params.get("device", "cpu"),
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNv2Classifier":
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


class TabPFNv2Regressor(RegressionModel):
    """TabPFN v2 regressor."""

    supports_tuning = False

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 8,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            name="TabPFN_v2",
            tuned=False,
            n_estimators=n_estimators,
            device=device,
            **kwargs,
        )

    def _create_model(self, **params):
        try:
            from tabpfn_v2 import TabPFNRegressor as TabPFNv2Reg
            return TabPFNv2Reg(
                n_estimators=self.params.get("n_estimators", 8),
                device=self.params.get("device", "cpu"),
            )
        except ImportError:
            # Placeholder - TabPFN v1 doesn't support regression
            raise NotImplementedError("TabPFN v2 regressor not available")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNv2Regressor":
        self.model = self._create_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


# =============================================================================
# TabPFN v2.5
# =============================================================================

class TabPFNv25Classifier(ClassificationModel):
    """TabPFN v2.5 classifier with latest improvements."""

    supports_tuning = False

    def __init__(
        self,
        tuned: bool = False,
        n_estimators: int = 16,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            name="TabPFN_v2.5",
            tuned=False,
            n_estimators=n_estimators,
            device=device,
            **kwargs,
        )

    def _create_model(self, **params):
        try:
            from tabpfn_v25 import TabPFNClassifier as TabPFNv25
            return TabPFNv25(
                n_estimators=self.params.get("n_estimators", 16),
                device=self.params.get("device", "cpu"),
            )
        except ImportError:
            # Fallback to v2 or v1
            try:
                from tabpfn_v2 import TabPFNClassifier as TabPFNv2
                return TabPFNv2(
                    n_estimators=self.params.get("n_estimators", 16),
                    device=self.params.get("device", "cpu"),
                )
            except ImportError:
                from tabpfn import TabPFNClassifier
                return TabPFNClassifier(
                    N_ensemble_configurations=self.params.get("n_estimators", 16) * 2,
                    device=self.params.get("device", "cpu"),
                )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNv25Classifier":
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


# =============================================================================
# Explainable Boosting Machine (EBM)
# =============================================================================

class EBMClassifier(ClassificationModel):
    """Explainable Boosting Machine classifier from InterpretML."""

    supports_tuning = True
    tuning_param_space = {
        "max_bins": {"type": "int", "low": 128, "high": 512},
        "max_interaction_bins": {"type": "int", "low": 16, "high": 64},
        "interactions": {"type": "int", "low": 0, "high": 20},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    }

    def __init__(
        self,
        tuned: bool = False,
        max_bins: int = 256,
        interactions: int = 10,
        learning_rate: float = 0.01,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="EBM",
            tuned=tuned,
            max_bins=max_bins,
            interactions=interactions,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from interpret.glassbox import ExplainableBoostingClassifier
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return ExplainableBoostingClassifier(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EBMClassifier":
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


class EBMRegressor(RegressionModel):
    """Explainable Boosting Machine regressor from InterpretML."""

    supports_tuning = True
    tuning_param_space = {
        "max_bins": {"type": "int", "low": 128, "high": 512},
        "interactions": {"type": "int", "low": 0, "high": 20},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        max_bins: int = 256,
        interactions: int = 10,
        learning_rate: float = 0.01,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            name="EBM",
            tuned=tuned,
            max_bins=max_bins,
            interactions=interactions,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _create_model(self, **params):
        from interpret.glassbox import ExplainableBoostingRegressor
        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)
        return ExplainableBoostingRegressor(**merged_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EBMRegressor":
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
# Placeholder models for advanced architectures
# These are stubs that can be implemented when packages become available
# =============================================================================

class RealMLPClassifier(ClassificationModel):
    """RealMLP classifier - placeholder for the RealMLP architecture."""

    supports_tuning = True
    tuning_param_space = {
        "hidden_size": {"type": "int", "low": 64, "high": 512},
        "num_layers": {"type": "int", "low": 2, "high": 8},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(
            name="RealMLP",
            tuned=tuned,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            **kwargs,
        )

    def _create_model(self, **params):
        # Placeholder - implement with actual RealMLP when available
        # For now, fallback to sklearn MLPClassifier
        from sklearn.neural_network import MLPClassifier
        merged_params = {**self.params, **params}
        hidden_size = merged_params.get("hidden_size", 256)
        num_layers = merged_params.get("num_layers", 4)
        hidden_layer_sizes = tuple([hidden_size] * num_layers)
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=merged_params.get("dropout", 0.1),
            learning_rate_init=merged_params.get("learning_rate", 1e-3),
            max_iter=merged_params.get("epochs", 100),
            batch_size=merged_params.get("batch_size", 256),
            random_state=merged_params.get("random_state", 42),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RealMLPClassifier":
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


class TabDPTClassifier(ClassificationModel):
    """TabDPT (Tabular Decision PreTraining) classifier."""

    supports_tuning = False  # Uses pretrained model

    def __init__(
        self,
        tuned: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            name="TabDPT",
            tuned=False,
            device=device,
            **kwargs,
        )

    def _create_model(self, **params):
        # Placeholder - implement when TabDPT package is available
        raise NotImplementedError("TabDPT not yet available. Install tabdpt package.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabDPTClassifier":
        try:
            self.model = self._create_model()
            self.model.fit(X, y)
        except NotImplementedError:
            # Fallback to TabPFN
            from tabpfn import TabPFNClassifier
            self.model = TabPFNClassifier(device=self.params.get("device", "cpu"))
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


class TabMClassifier(ClassificationModel):
    """TabM (Tabular Mamba) classifier."""

    supports_tuning = True
    tuning_param_space = {
        "d_model": {"type": "int", "low": 32, "high": 256},
        "n_layers": {"type": "int", "low": 2, "high": 8},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    }

    def __init__(
        self,
        tuned: bool = False,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 256,
        random_state: Optional[int] = 42,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            name="TabM",
            tuned=tuned,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
            **kwargs,
        )

    def _create_model(self, **params):
        # Placeholder - implement when TabM package is available
        # Fallback to MLP
        from sklearn.neural_network import MLPClassifier
        merged_params = {**self.params, **params}
        d_model = merged_params.get("d_model", 128)
        n_layers = merged_params.get("n_layers", 4)
        hidden_layer_sizes = tuple([d_model] * n_layers)
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=merged_params.get("epochs", 100),
            random_state=merged_params.get("random_state", 42),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabMClassifier":
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


class ModernNCAClassifier(ClassificationModel):
    """Modern Neighborhood Component Analysis classifier."""

    supports_tuning = True
    tuning_param_space = {
        "n_components": {"type": "int", "low": 2, "high": 100},
        "max_iter": {"type": "int", "low": 50, "high": 500},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_components: Optional[int] = None,
        max_iter: int = 100,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(
            name="ModernNCA",
            tuned=tuned,
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        from sklearn.pipeline import Pipeline
        from sklearn.neighbors import KNeighborsClassifier

        merged_params = {**self.params, **params}
        merged_params.pop("tuned", None)
        merged_params.pop("n_trials", None)
        merged_params.pop("validation_split", None)

        nca = NeighborhoodComponentsAnalysis(
            n_components=merged_params.get("n_components"),
            max_iter=merged_params.get("max_iter", 100),
            random_state=merged_params.get("random_state", 42),
        )
        knn = KNeighborsClassifier(n_neighbors=5)
        return Pipeline([("nca", nca), ("knn", knn)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModernNCAClassifier":
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


class xRFMClassifier(ClassificationModel):
    """xRFM (Extended Random Feature Maps) classifier."""

    supports_tuning = True
    tuning_param_space = {
        "n_components": {"type": "int", "low": 100, "high": 2000},
        "gamma": {"type": "float", "low": 1e-4, "high": 1e2, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        n_components: int = 500,
        gamma: float = 1.0,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(
            name="xRFM",
            tuned=tuned,
            n_components=n_components,
            gamma=gamma,
            random_state=random_state,
            **kwargs,
        )

    def _create_model(self, **params):
        from sklearn.kernel_approximation import RBFSampler
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline

        merged_params = {**self.params, **params}

        rbf = RBFSampler(
            n_components=merged_params.get("n_components", 500),
            gamma=merged_params.get("gamma", 1.0),
            random_state=merged_params.get("random_state", 42),
        )
        clf = SGDClassifier(
            loss="log_loss",
            random_state=merged_params.get("random_state", 42),
        )
        return Pipeline([("rbf", rbf), ("clf", clf)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "xRFMClassifier":
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


class TorchMLPClassifier(ClassificationModel):
    """PyTorch MLP classifier with modern training techniques."""

    supports_tuning = True
    tuning_param_space = {
        "hidden_size": {"type": "int", "low": 64, "high": 512},
        "num_layers": {"type": "int", "low": 2, "high": 6},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        random_state: Optional[int] = 42,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            name="TorchMLP",
            tuned=tuned,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
            **kwargs,
        )

    def _create_model(self, **params):
        # This will be fully implemented in deep_learning.py
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import LabelEncoder

        merged_params = {**self.params}
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            merged_params.update(self._best_params)

        # Set seed
        if merged_params.get("random_state"):
            torch.manual_seed(merged_params["random_state"])

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        n_classes = len(self._label_encoder.classes_)

        # Build model
        hidden_size = merged_params.get("hidden_size", 256)
        num_layers = merged_params.get("num_layers", 3)
        dropout = merged_params.get("dropout", 0.2)

        layers = []
        input_size = X.shape[1]
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_size, n_classes))

        self.model = nn.Sequential(*layers)
        device = torch.device(merged_params.get("device", "cpu"))
        self.model = self.model.to(device)

        # Training
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y_encoded).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=merged_params.get("batch_size", 64),
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=merged_params.get("learning_rate", 1e-3),
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(merged_params.get("epochs", 100)):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self._device = device
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return self._label_encoder.inverse_transform(predicted.cpu().numpy())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()


class SAPRPTClassifier(ClassificationModel):
    """SAP-RPT-OSS classifier - Self-Attention Pretrained model."""

    supports_tuning = False  # Pretrained model

    def __init__(
        self,
        tuned: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            name="SAP-RPT-OSS",
            tuned=False,
            device=device,
            **kwargs,
        )

    def _create_model(self, **params):
        # Placeholder - implement when SAP-RPT-OSS is available
        raise NotImplementedError("SAP-RPT-OSS not yet available")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAPRPTClassifier":
        try:
            self.model = self._create_model()
            self.model.fit(X, y)
        except NotImplementedError:
            # Fallback to TabPFN
            from tabpfn import TabPFNClassifier
            self.model = TabPFNClassifier(device=self.params.get("device", "cpu"))
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

