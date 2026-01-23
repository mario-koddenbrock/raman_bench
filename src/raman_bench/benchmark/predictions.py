"""
Prediction computation for the benchmark pipeline.
"""
import logging
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from raman_bench.benchmark.model import AutoGluonModel
from raman_bench.benchmark.preprocessing import get_preprocessing_pipeline, handle_missing_values
from raman_data import raman_data, TASK_TYPE

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def compute_predictions(config):
    logger.info("\n" + "=" * 60 + "\nSTEP 1: Computing Predictions")
    output_dir = config["output_dir"]
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    classification_datasets = config.get("classification_datasets", [])
    classification_models = config.get("classification_models", [])

    if classification_datasets and classification_models:
        logger.info("\n--- Classification Tasks ---")
        _compute_predictions_for_task(
            datasets=classification_datasets,
            model_configs=classification_models,
            task_type=TASK_TYPE.Classification,
            config=config,
            predictions_dir=predictions_dir,
        )

    regression_datasets = config.get("regression_datasets", [])
    regression_models = config.get("regression_models", [])
    if regression_datasets and regression_models:
        logger.info("\n--- Regression Tasks ---")
        _compute_predictions_for_task(
            datasets=regression_datasets,
            model_configs=regression_models,
            task_type=TASK_TYPE.Regression,
            config=config,
            predictions_dir=predictions_dir,
        )
    logger.info(f"\nPredictions saved to: {predictions_dir}")

def _compute_predictions_for_task(datasets, model_configs, task_type, config, pipeline):

    test_size = config.get("test_size", 0.2) # TODO
    random_state = config.get("random_state", 42)
    total = len(datasets) * len(model_configs)
    pbar = tqdm(total=total, desc=f"Computing {task_type} predictions")
    predictions_dir = config.get("predictions_dir", "results/predictions")

    for dataset_name in datasets:
        dataset = raman_data(dataset_name)
        logger.info(f"Computing predictions for dataset: {dataset_name}")
        logger.info(f"Dataset size: {len(dataset)} samples")

        if len(dataset) < 10:
            logger.warning("Dataset size is small. Skipping predictions computation.")
            continue

        # TODO Data augmentation

        if pipeline is not None:
            dataset = pipeline.transform_dataset(dataset)

        num_targets = dataset.targets.shape[1] if dataset.targets.ndim > 1 else 1

        for target_idx in range(num_targets):

            data_df = dataset.to_dataframe(target_idx)
            data_df = handle_missing_values(config, data_df)

            data_train, data_test = train_test_split(data_df, test_size=test_size, random_state=random_state)

            for model_config in model_configs:

                model_name = model_config["name"]
                pbar.set_description(f"{dataset_name} | {model_name}")

                model = AutoGluonModel(model_config)
                model.fit(data_train)
                y_pred = model.predict(data_test)

                safe_dataset = dataset_name.replace("/", "_").replace("\\", "_")
                filename = f"{safe_dataset}_{target_idx}_{model_name}_predictions.csv"

                y_pred.sort_index().to_csv(os.path.join(predictions_dir, filename), index=True)

        pbar.update(1)
        pbar.close()
