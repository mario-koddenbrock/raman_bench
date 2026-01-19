"""
Deep learning models for Raman spectroscopy.

Includes MLP and 1D CNN implementations using PyTorch.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from raman_bench.models.base import ClassificationModel, RegressionModel


class MLPClassifier(ClassificationModel):
    """Multi-Layer Perceptron classifier for Raman spectroscopy."""

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
        hidden_layers: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: Optional[int] = 42,
        device: str = "cpu",
        **kwargs,
    ):
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]
        super().__init__(
            name="MLP",
            tuned=tuned,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
            **kwargs,
        )

    def _create_model(self, input_dim: int, output_dim: int, **params) -> "torch.nn.Module":
        """Build the MLP model."""
        import torch.nn as nn

        merged_params = {**self.params, **params}
        hidden_layers = merged_params.get("hidden_layers", [256, 128, 64])
        dropout = merged_params.get("dropout", 0.3)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        """Fit the model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import LabelEncoder

        merged_params = dict(self.params)
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            merged_params.update(self._best_params)

        # Set random seed
        if merged_params.get("random_state"):
            torch.manual_seed(merged_params["random_state"])

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        n_classes = len(self._label_encoder.classes_)

        # Prepare data
        device = torch.device(merged_params.get("device", "cpu"))
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y_encoded).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=merged_params.get("batch_size", 32),
            shuffle=True,
        )

        # Build model
        self.model = self._create_model(X.shape[1], n_classes, **merged_params)
        self.model = self.model.to(device)

        # Setup training
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=merged_params.get("learning_rate", 0.001),
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
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
        """Make predictions."""
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
        """Predict class probabilities."""
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


class MLPRegressor(RegressionModel):
    """Multi-Layer Perceptron regressor for Raman spectroscopy."""

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
        hidden_layers: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: Optional[int] = 42,
        device: str = "cpu",
        **kwargs,
    ):
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]
        super().__init__(
            name="MLP",
            tuned=tuned,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
            **kwargs,
        )

    def _create_model(self, input_dim: int, **params) -> "torch.nn.Module":
        """Build the MLP model."""
        import torch.nn as nn

        merged_params = {**self.params, **params}
        hidden_layers = merged_params.get("hidden_layers", [256, 128, 64])
        dropout = merged_params.get("dropout", 0.3)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        """Fit the model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        merged_params = dict(self.params)
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            merged_params.update(self._best_params)

        if merged_params.get("random_state"):
            torch.manual_seed(merged_params["random_state"])

        device = torch.device(merged_params.get("device", "cpu"))
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=merged_params.get("batch_size", 32),
            shuffle=True,
        )

        self.model = self._create_model(X.shape[1], **merged_params)
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=merged_params.get("learning_rate", 0.001),
        )
        criterion = nn.MSELoss()

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
        """Make predictions."""
        import torch

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            outputs = self.model(X_tensor)

        return outputs.cpu().numpy().flatten()


class CNN1DClassifier(ClassificationModel):
    """1D Convolutional Neural Network classifier for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_filters": {"type": "int", "low": 16, "high": 128},
        "kernel_size": {"type": "int", "low": 3, "high": 15},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        fc_layers: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: Optional[int] = 42,
        device: str = "cpu",
        **kwargs,
    ):
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        if fc_layers is None:
            fc_layers = [256, 128]
        super().__init__(
            name="CNN1D",
            tuned=tuned,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            fc_layers=fc_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
            **kwargs,
        )

    def _create_model(self, input_dim: int, output_dim: int, **params) -> "torch.nn.Module":
        """Build the CNN model."""
        import torch.nn as nn

        merged_params = {**self.params, **params}
        conv_channels = merged_params.get("conv_channels", [32, 64, 128])
        kernel_sizes = merged_params.get("kernel_sizes", [7, 5, 3])
        fc_layers = merged_params.get("fc_layers", [256, 128])
        dropout = merged_params.get("dropout", 0.3)

        class CNN1D(nn.Module):
            def __init__(self):
                super().__init__()

                # Convolutional layers
                conv_list = []
                in_channels = 1
                current_length = input_dim

                for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
                    conv_list.extend([
                        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                        nn.ReLU(),
                        nn.BatchNorm1d(out_channels),
                        nn.MaxPool1d(2),
                        nn.Dropout(dropout),
                    ])
                    in_channels = out_channels
                    current_length = current_length // 2

                self.conv = nn.Sequential(*conv_list)

                # Calculate flattened size
                flat_size = conv_channels[-1] * current_length

                # Fully connected layers
                fc = []
                prev_size = flat_size

                for fc_size in fc_layers:
                    fc.extend([
                        nn.Linear(prev_size, fc_size),
                        nn.ReLU(),
                        nn.BatchNorm1d(fc_size),
                        nn.Dropout(dropout),
                    ])
                    prev_size = fc_size

                fc.append(nn.Linear(prev_size, output_dim))
                self.fc = nn.Sequential(*fc)

            def forward(self, x):
                # Add channel dimension
                x = x.unsqueeze(1)
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        return CNN1D()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNN1DClassifier":
        """Fit the model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import LabelEncoder

        merged_params = dict(self.params)
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            merged_params.update(self._best_params)

        if merged_params.get("random_state"):
            torch.manual_seed(merged_params["random_state"])

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        n_classes = len(self._label_encoder.classes_)

        device = torch.device(merged_params.get("device", "cpu"))
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y_encoded).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=merged_params.get("batch_size", 32),
            shuffle=True,
        )

        self.model = self._create_model(X.shape[1], n_classes, **merged_params)
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=merged_params.get("learning_rate", 0.001),
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
        """Make predictions."""
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
        """Predict class probabilities."""
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


class CNN1DRegressor(RegressionModel):
    """1D Convolutional Neural Network regressor for Raman spectroscopy."""

    supports_tuning = True
    tuning_param_space = {
        "n_filters": {"type": "int", "low": 16, "high": 128},
        "kernel_size": {"type": "int", "low": 3, "high": 15},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    }

    def __init__(
        self,
        tuned: bool = False,
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        fc_layers: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: Optional[int] = 42,
        device: str = "cpu",
        **kwargs,
    ):
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        if fc_layers is None:
            fc_layers = [256, 128]
        super().__init__(
            name="CNN1D",
            tuned=tuned,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            fc_layers=fc_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
            **kwargs,
        )

    def _create_model(self, input_dim: int, **params) -> "torch.nn.Module":
        """Build the CNN model."""
        import torch.nn as nn

        merged_params = {**self.params, **params}
        conv_channels = merged_params.get("conv_channels", [32, 64, 128])
        kernel_sizes = merged_params.get("kernel_sizes", [7, 5, 3])
        fc_layers = merged_params.get("fc_layers", [256, 128])
        dropout = merged_params.get("dropout", 0.3)

        class CNN1D(nn.Module):
            def __init__(self):
                super().__init__()

                conv_list = []
                in_channels = 1
                current_length = input_dim

                for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
                    conv_list.extend([
                        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                        nn.ReLU(),
                        nn.BatchNorm1d(out_channels),
                        nn.MaxPool1d(2),
                        nn.Dropout(dropout),
                    ])
                    in_channels = out_channels
                    current_length = current_length // 2

                self.conv = nn.Sequential(*conv_list)

                flat_size = conv_channels[-1] * current_length

                fc = []
                prev_size = flat_size

                for fc_size in fc_layers:
                    fc.extend([
                        nn.Linear(prev_size, fc_size),
                        nn.ReLU(),
                        nn.BatchNorm1d(fc_size),
                        nn.Dropout(dropout),
                    ])
                    prev_size = fc_size

                fc.append(nn.Linear(prev_size, 1))
                self.fc = nn.Sequential(*fc)

            def forward(self, x):
                x = x.unsqueeze(1)
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        return CNN1D()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNN1DRegressor":
        """Fit the model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        merged_params = dict(self.params)
        if self.tuned and self.supports_tuning:
            self._best_params = self._tune_hyperparameters(X, y)
            merged_params.update(self._best_params)

        if merged_params.get("random_state"):
            torch.manual_seed(merged_params["random_state"])

        device = torch.device(merged_params.get("device", "cpu"))
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=merged_params.get("batch_size", 32),
            shuffle=True,
        )

        self.model = self._create_model(X.shape[1], **merged_params)
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=merged_params.get("learning_rate", 0.001),
        )
        criterion = nn.MSELoss()

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
        """Make predictions."""
        import torch

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            outputs = self.model(X_tensor)

        return outputs.cpu().numpy().flatten()

