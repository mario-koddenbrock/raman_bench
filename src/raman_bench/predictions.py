"""
Prediction computation for the benchmark pipeline.
"""
import logging
import os

from tqdm import tqdm

from raman_bench.benchmark import RamanBenchmark
from raman_bench.model import AutoGluonModel
from raman_data import TASK_TYPE

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def compute_predictions(config):
    logger.info("\n" + "=" * 60 + "\nSTEP 1: Computing Predictions")

    output_dir = config["output_dir"]
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    n_classification = config["n_classification"]
    n_regression = config["n_regression"]
    test_size = config["test_size"]
    random_state = config["random_state"]
    preprocessing = config["preprocessing"]
    augmentation = config["augmentation"]
    cache_dir = config["cache_dir"]

    benchmark = RamanBenchmark(
        n_classification=n_classification,
        n_regression=n_regression,
        test_size=test_size,
        random_state=random_state,
        preprocessing=preprocessing,
        augmentation=augmentation,
        cache_dir=cache_dir,
    )

    model_configs = config["model_configs"]
    autogluon_time_limit = config["autogluon_time_limit"]
    autogluon_presets = config["autogluon_presets"]

    pbar = tqdm(total=len(benchmark)*len(model_configs))

    for data_train, data_test, key, task_type in benchmark:
        for model_config in model_configs:

            model_name = model_config["name"]
            pbar.set_description(f"{key} | {model_name}")

            model = AutoGluonModel(
                model_config=model_config,
                task_type=task_type,
                autogluon_time_limit=autogluon_time_limit,
                autogluon_presets=autogluon_presets,
                autogluon_path=os.path.join(cache_dir, "autogluon", key),
            )

            model.fit(data_train)
            y_pred = model.predict(data_test)

            filename = f"{key}_{model_name}_predictions.csv"
            y_pred.sort_index().to_csv(os.path.join(predictions_dir, filename), index=True)

            pbar.update(1)
    pbar.close()
