"""
Base model classes for Raman Bench.

Defines the interface that all models must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all Raman Bench models.

    All models must implement fit, predict, and get_params methods.
    """

    # Class attribute indicating if this model supports HPO tuning
    supports_tuning: bool = False
    # Default hyperparameter search space for tuning
    tuning_param_space: Dict[str, Any] = {}

    def __init__(
        self,
        name: str,
        task_type: str = "classification",
        tuned: bool = False,
        n_trials: int = 50,
        validation_split: float = 0.2,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        """
        Initialize the base model.

        Args:
            name: Model name/identifier
            task_type: Type of task ('classification' or 'regression')
            tuned: Whether to perform HPO tuning
            n_trials: Number of HPO trials (if tuned=True)
            validation_split: Fraction of training data for validation during HPO
            random_state: Random seed for reproducibility
            **kwargs: Additional model parameters
        """
        self.name = name
        self.task_type = task_type
        self.tuned = tuned
        self.n_trials = n_trials
        self.validation_split = validation_split
        self.random_state = random_state
        self.params = kwargs
        self.model = None
        self._is_fitted = False
        self._best_params = None

    @abstractmethod
    def _create_model(self, **params):
        """
        Create the underlying model with given parameters.

        Args:
            **params: Model parameters

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        Fit the model to training data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predictions array
        """
        pass

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (for classification models).

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Probability predictions or None if not supported
        """
        return None

    def _tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Best hyperparameters found
        """
        import optuna
        from sklearn.model_selection import train_test_split

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Split data for validation
        stratify = y if self.task_type == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=stratify,
        )

        def objective(trial):
            # Sample hyperparameters
            params = self._sample_params(trial)

            # Create and train model
            model = self._create_model(**params)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)

            if self.task_type == "classification":
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_val, y_pred)
            else:
                from sklearn.metrics import r2_score
                return r2_score(y_val, y_pred)

        # Run optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        return study.best_params

    def _sample_params(self, trial) -> Dict[str, Any]:
        """
        Sample hyperparameters from the search space.

        Args:
            trial: Optuna trial object

        Returns:
            Sampled parameters dictionary
        """
        params = {}
        for param_name, param_config in self.tuning_param_space.items():
            param_type = param_config.get("type", "categorical")

            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )

        return params

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of parameters
        """
        result = {
            "name": self.name,
            "task_type": self.task_type,
            "tuned": self.tuned,
            **self.params,
        }
        if self._best_params:
            result["best_params"] = self._best_params
        return result

    def set_params(self, **params) -> "BaseModel":
        """
        Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            self
        """
        for key, value in params.items():
            if key in ["name", "task_type", "tuned"]:
                setattr(self, key, value)
            else:
                self.params[key] = value
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    @property
    def display_name(self) -> str:
        """Get display name including tuned suffix if applicable."""
        if self.tuned:
            return f"{self.name} (Tuned)"
        return self.name

    def __repr__(self) -> str:
        """String representation."""
        tuned_str = " [Tuned]" if self.tuned else ""
        return f"{self.__class__.__name__}(name='{self.name}'{tuned_str}, task_type='{self.task_type}')"


class ClassificationModel(BaseModel):
    """Base class for classification models."""

    def __init__(self, name: str, tuned: bool = False, **kwargs):
        """Initialize classification model."""
        super().__init__(name=name, task_type="classification", tuned=tuned, **kwargs)


class RegressionModel(BaseModel):
    """Base class for regression models."""

    def __init__(self, name: str, tuned: bool = False, **kwargs):
        """Initialize regression model."""
        super().__init__(name=name, task_type="regression", tuned=tuned, **kwargs)

