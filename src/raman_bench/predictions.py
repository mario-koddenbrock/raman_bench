"""
Prediction computation for the benchmark pipeline.
"""
import logging
import os

from tqdm import tqdm

from raman_bench.benchmark import RamanBenchmark, configure_benchmark
from raman_bench.model import AutoGluonModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def compute_predictions(config):
    logger.info("\n" + "=" * 60 + "\nSTEP 1: Computing Predictions")

    output_dir = config["output_dir"]
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    cache_dir = config["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    benchmark = configure_benchmark(config)
    benchmark.init_datasets()

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
