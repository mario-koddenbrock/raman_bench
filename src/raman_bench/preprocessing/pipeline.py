"""
Preprocessing pipeline implementation using RamanSPy.
"""

from typing import Any, Dict, List, Optional, Union
from raman_data import TASK_TYPE, raman_data, RamanDataset
import numpy as np



class PreprocessingPipeline:
    """
    Preprocessing pipeline for Raman spectroscopy data.

    Wraps RamanSPy preprocessing methods in a scikit-learn compatible pipeline.
    """

    def __init__(
        self,
        steps: Optional[List[Any]] = None,
        name: str = "default",
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            steps: List of RamanSPy preprocessing steps
            name: Name identifier for the pipeline
        """
        self.steps = steps or []
        self.name = name
        self._ramanspy_pipeline = None

    def _build_pipeline(self):
        """Build the RamanSPy pipeline from steps."""
        import ramanspy as rp

        if self.steps:
            self._ramanspy_pipeline = rp.preprocessing.Pipeline(self.steps)
        else:
            self._ramanspy_pipeline = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PreprocessingPipeline":
        """
        Fit the pipeline (no-op for most preprocessing steps).

        Args:
            X: Input spectral data (n_samples, n_features)
            y: Target values (unused)

        Returns:
            self
        """
        self._build_pipeline()
        return self

    def transform(
        self,
        X: np.ndarray,
        wavenumbers: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Transform the input data using the preprocessing pipeline.

        Args:
            X: Input spectral data (n_samples, n_features)
            wavenumbers: Wavenumber/shift values

        Returns:
            Preprocessed spectral data
        """
        import ramanspy as rp

        if self._ramanspy_pipeline is None:
            self._build_pipeline()

        if self._ramanspy_pipeline is None:
            return X

        # Convert to RamanSPy format
        if wavenumbers is None:
            wavenumbers = np.arange(X.shape[1])

        # Process each spectrum
        processed_spectra = []
        for i in range(X.shape[0]):
            # Create a RamanSPy Spectrum object
            spectrum = rp.Spectrum(X[i], wavenumbers)
            # Apply preprocessing
            processed = self._ramanspy_pipeline.apply(spectrum)
            processed_spectra.append(processed.spectral_data)

        return np.array(processed_spectra)

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        wavenumbers: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit and transform the input data.

        Args:
            X: Input spectral data (n_samples, n_features)
            y: Target values (unused)
            wavenumbers: Wavenumber/shift values

        Returns:
            Preprocessed spectral data
        """
        self.fit(X, y)
        return self.transform(X, wavenumbers)

    def transform_dataset(self, dataset: RamanDataset) -> RamanDataset:
        """
        Transform a RamanDataset.

        Args:
            dataset: Input dataset

        Returns:
            New dataset with preprocessed data
        """
        processed_data = self.transform(dataset.spectra, dataset.raman_shifts)
        dataset.spectra = processed_data
        dataset.metadata["preprocessed"] = True
        dataset.metadata["pipeline"] = self.name

        return dataset

    def add_step(self, step: Any) -> "PreprocessingPipeline":
        """
        Add a preprocessing step to the pipeline.

        Args:
            step: RamanSPy preprocessing step

        Returns:
            self
        """
        self.steps.append(step)
        self._ramanspy_pipeline = None  # Reset pipeline
        return self

    def get_params(self) -> Dict[str, Any]:
        """
        Get pipeline parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "name": self.name,
            "n_steps": len(self.steps),
            "steps": [str(step) for step in self.steps],
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"PreprocessingPipeline(name='{self.name}', n_steps={len(self.steps)})"


class IdentityPipeline(PreprocessingPipeline):
    """
    Identity pipeline that returns data unchanged.

    Useful for baseline comparisons or when preprocessing is handled elsewhere.
    """

    def __init__(self):
        """Initialize identity pipeline."""
        super().__init__(steps=[], name="identity")

    def transform(
        self,
        X: np.ndarray,
        wavenumbers: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return data unchanged."""
        return X

