"""
Regression metrics for Raman Bench.

Provides comprehensive evaluation metrics for regression tasks.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, max_error, median_absolute_error, r2_score, \
    mean_absolute_error


class RegressionMetrics:
    """
    Regression metrics calculator.

    Computes various metrics for evaluating regression models.
    """

    def __init__(self):
        """Initialize regression metrics."""
        pass

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        metrics["mse"] = self.mse(y_true, y_pred)
        metrics["rmse"] = self.rmse(y_true, y_pred)
        metrics["mae"] = self.mae(y_true, y_pred)
        metrics["r2"] = self.r2(y_true, y_pred)
        metrics["mape"] = self.mape(y_true, y_pred)
        metrics["explained_variance"] = self.explained_variance(y_true, y_pred)
        metrics["max_error"] = self.max_error(y_true, y_pred)
        metrics["median_ae"] = self.median_ae(y_true, y_pred)

        return metrics

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        return float(mean_squared_error(y_true, y_pred))

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        return float(np.sqrt(self.mse(y_true, y_pred)))

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return float(mean_absolute_error(y_true, y_pred))

    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² (coefficient of determination)."""
        return float(r2_score(y_true, y_pred))

    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Percentage Error.

        Note: Returns 0 if y_true contains zeros to avoid division by zero.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return 0.0

        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Explained Variance Score."""
        return float(explained_variance_score(y_true, y_pred))

    def max_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Maximum Error."""
        return float(max_error(y_true, y_pred))

    def median_ae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Median Absolute Error."""
        return float(median_absolute_error(y_true, y_pred))

    def huber_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        delta: float = 1.0,
    ) -> float:
        """
        Compute Huber Loss.

        Args:
            y_true: True values
            y_pred: Predicted values
            delta: Threshold for switching between MSE and MAE

        Returns:
            Huber loss value
        """
        residual = np.abs(y_true - y_pred)
        mask = residual <= delta

        loss = np.where(
            mask,
            0.5 * residual ** 2,
            delta * residual - 0.5 * delta ** 2,
        )

        return float(np.mean(loss))

    def adjusted_r2(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: int,
    ) -> float:
        """
        Compute Adjusted R².

        Args:
            y_true: True values
            y_pred: Predicted values
            n_features: Number of features used in the model

        Returns:
            Adjusted R² score
        """
        n_samples = len(y_true)
        r2 = self.r2(y_true, y_pred)

        if n_samples <= n_features + 1:
            return r2

        adjusted = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        return float(adjusted)

    def get_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get residual analysis statistics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with residual statistics
        """
        residuals = y_true - y_pred

        return {
            "residuals": residuals,
            "mean": np.mean(residuals),
            "std": np.std(residuals),
            "min": np.min(residuals),
            "max": np.max(residuals),
            "median": np.median(residuals),
            "q1": np.percentile(residuals, 25),
            "q3": np.percentile(residuals, 75),
        }

