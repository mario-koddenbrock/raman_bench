import logging
from typing import Any, Dict, Optional

from autogluon.common import TabularDataset
from autogluon.tabular import TabularPredictor
from pandas import DataFrame

logger = logging.getLogger(__name__)


class AutoGluonModel:
    """Wrapper around AutoGluon's TabularPredictor to expose a simple create/fit/predict API.

    Config keys supported (all optional, defaults shown):
      - label: str = 'target'
      - metric: str = 'rmse'
      - time_limit: int = 7200
      - presets: str = 'best_quality'

    Usage:
      model = AutoGluonModel(config)
      model.create()
      model.fit(train_df)
      preds = model.predict(test_df)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.label: str = self.config.get("label", "target") # TODO
        self.metric: str = self.config.get("metric", "rmse") # TODO
        self.time_limit: int = int(self.config.get("time_limit", 120)) # TODO
        self.presets: Any = self.config.get("presets", "best_quality") # TODO

        self.predictor: TabularPredictor = TabularPredictor(label=self.label, eval_metric=self.metric)
        logger.debug("Initialized AutoGluonModel with label=%s metric=%s time_limit=%s presets=%s",
                     self.label, self.metric, self.time_limit, self.presets)

    def fit(self, data_train: DataFrame) -> TabularPredictor:
        """Fit the predictor on the provided training data.

        data_train may be a pandas DataFrame, a path, or any input accepted by TabularDataset.
        """
        if self.predictor is None:
            logger.debug("Predictor not created yet, creating before fit")
            self.create()

        tabular_data = TabularDataset(data_train)
        logger.info("Starting fit: rows=%s time_limit=%s presets=%s", len(tabular_data), self.time_limit, self.presets)

        self.predictor.fit(tabular_data, time_limit=self.time_limit, presets=self.presets)
        logger.info("Fit completed")
        return self.predictor

    def predict(self, data_test: DataFrame):
        """Run prediction using the fitted predictor and return predictions.

        Raises:
            RuntimeError: if the predictor has not been created/fitted yet.
        """
        if self.predictor is None:
            raise RuntimeError("Predictor is not initialized. Call create() and fit() before predict().")

        # If input isn't already a TabularDataset, wrap it.
        test_input = data_test if isinstance(data_test, TabularDataset) else TabularDataset(data_test)
        logger.info("Generating predictions for %s rows", len(test_input))
        predictions = self.predictor.predict(test_input)
        return predictions

