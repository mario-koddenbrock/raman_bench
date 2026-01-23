import logging
from typing import Any, List

from autogluon.common import TabularDataset
from autogluon.tabular import TabularPredictor
from pandas import DataFrame

from raman_data import TASK_TYPE

logger = logging.getLogger(__name__)


class AutoGluonModel:
    """Wrapper around AutoGluon's TabularPredictor to expose a simple create/fit/predict API.

    Config keys supported (all optional, defaults shown):
      - label: str = 'target'
      - metric: str = 'rmse'
      - time_limit: int = 7200
      - presets: str = 'best_quality'
      - models: list[str] = ['LGBModel', 'CatBoostModel', 'XGBoostModel', 'RealMLPModel', 'TabMModel', 'MitraModel', 'TabICLModel', 'TabPFNV2Model', 'RFModel', 'XTModel', 'KNNModel', 'LinearModel', 'TabularNeuralNetTorchModel', 'NNFastAiTabularModel']
      - ensemble: bool = True
      - optimize: bool = True

    Usage:
      model = AutoGluonModel(config)
      model.create()
      model.fit(train_df)
      preds = model.predict(test_df)
    """

    def __init__(
            self,
            models: List[str],
            ensemble: bool = True,
            optimize: bool = True,
            task_type: TASK_TYPE = TASK_TYPE.Regression,
            autogluon_time_limit: int = 60,
            autogluon_presets: str = "best_quality",
            autogluon_path: str = None,
    ) -> None:

        self.models = models
        self.label: str = "target"
        self.autogluon_path = autogluon_path

        if task_type == TASK_TYPE.Regression:
            self.metric: str = "rmse"
        elif task_type == TASK_TYPE.Classification:
            self.metric: str = "balanced_accuracy" # TODO check if this is the best metric for classification
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        self.time_limit: int = autogluon_time_limit
        self.presets: Any = autogluon_presets
        self.ensemble: bool = ensemble
        self.optimize: bool = optimize

        self.predictor: TabularPredictor = TabularPredictor(
            label=self.label,
            eval_metric=self.metric,
            path=autogluon_path
        )
        logger.debug("Initialized AutoGluonModel with label=%s metric=%s time_limit=%s presets=%s models=%s ensemble=%s optimize=%s",
                     self.label, self.metric, self.time_limit, self.presets, self.models, self.ensemble, self.optimize)


    def fit(self, data_train: DataFrame) -> TabularPredictor:
        """Fit the predictor on the provided training data.

        data_train may be a pandas DataFrame, a path, or any input accepted by TabularDataset.
        """
        tabular_data = TabularDataset(data_train)
        logger.info("Starting fit: rows=%s time_limit=%s presets=%s models=%s ensemble=%s optimize=%s",
                    len(tabular_data), self.time_limit, self.presets, self.models, self.ensemble, self.optimize)

        fit_args = {
            "time_limit": self.time_limit,
            "presets": self.presets,
            "hyperparameters": {"models": self.models}
        }

        if not self.ensemble:
            fit_args["num_bag_folds"] = 0

        if not self.optimize:
            fit_args["hyperparameter_tune_kwargs"] = None

        self.predictor.fit(tabular_data, **fit_args)
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
