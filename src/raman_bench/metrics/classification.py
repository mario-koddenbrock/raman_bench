"""
Classification metrics for Raman Bench.

Provides comprehensive evaluation metrics for classification tasks.
"""

from typing import Dict, Optional, Union

import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, f1_score, \
    accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer


class ClassificationMetrics:
    """
    Classification metrics calculator.

    Computes various metrics for evaluating classification models.
    """

    def __init__(self, average: str = "weighted"):
        """
        Initialize classification metrics.

        Args:
            average: Averaging method for multi-class metrics
                    ('micro', 'macro', 'weighted', 'samples')
        """
        self.average = average

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC-AUC)

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = self.accuracy(y_true, y_pred)
        metrics["precision"] = self.precision(y_true, y_pred)
        metrics["recall"] = self.recall(y_true, y_pred)
        metrics["f1_score"] = self.f1_score(y_true, y_pred)

        # Additional metrics
        metrics["balanced_accuracy"] = self.balanced_accuracy(y_true, y_pred)
        metrics["cohen_kappa"] = self.cohen_kappa(y_true, y_pred)
        metrics["matthews_corrcoef"] = self.matthews_corrcoef(y_true, y_pred)

        # ROC-AUC if probabilities are available
        if y_proba is not None:
            try:
                metrics["roc_auc"] = self.roc_auc(y_true, y_proba)
            except Exception:
                metrics["roc_auc"] = np.nan

        return metrics

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return precision_score(y_true, y_pred, average=self.average, zero_division=0)

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return recall_score(y_true, y_pred, average=self.average, zero_division=0)

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average=self.average, zero_division=0)

    def balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(balanced_accuracy_score(y_true, y_pred))

    def cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(cohen_kappa_score(y_true, y_pred))

    def matthews_corrcoef(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(matthews_corrcoef(y_true, y_pred))

    def roc_auc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        multi_class: str = "ovr",
    ) -> float:
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            # Binary classification
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            return float(roc_auc_score(y_true, y_proba))
        else:
            # Multi-class classification
            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            return float(roc_auc_score(y_true_bin, y_proba, multi_class=multi_class, average=self.average))

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None,
    ) -> np.ndarray:
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = True,
    ) -> Union[str, Dict]:
        return classification_report(y_true, y_pred, output_dict=output_dict, zero_division=0)

    def get_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        report = self.classification_report(y_true, y_pred, output_dict=True)

        # Filter out summary statistics
        per_class = {}
        for key, value in report.items():
            if key not in ["accuracy", "macro avg", "weighted avg"]:
                per_class[key] = value

        return per_class

