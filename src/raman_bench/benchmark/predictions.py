"""
Prediction computation for the benchmark pipeline.
"""
import os
import logging
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from raman_bench.benchmark.model_factory import create_model
from raman_bench.benchmark.preprocessing import get_preprocessing_pipeline
from raman_data import raman_data, TASK_TYPE

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def compute_predictions(config):
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Computing Predictions")
    logger.info("=" * 60)
    output_dir = config["output_dir"]
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    pipeline = get_preprocessing_pipeline(config.get("preprocessing", "default"))
    classification_datasets = config.get("classification_datasets", [])
    classification_models = config.get("classification_models", [])
    if classification_datasets and classification_models:
        logger.info("\n--- Classification Tasks ---")
        _compute_predictions_for_task(
            datasets=classification_datasets,
            model_configs=classification_models,
            task_type=TASK_TYPE.Classification,
            config=config,
            pipeline=pipeline,
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
            pipeline=pipeline,
            predictions_dir=predictions_dir,
        )
    logger.info(f"\nPredictions saved to: {predictions_dir}")

def _compute_predictions_for_task(datasets, model_configs, task_type, config, pipeline, predictions_dir):
    cv_folds = config.get("cv_folds", 5)
    random_state = config.get("random_state", 42)
    total = len(datasets) * len(model_configs)
    pbar = tqdm(total=total, desc=f"Computing {task_type} predictions")
    for dataset_name in datasets:
        dataset = raman_data(dataset_name)
        if pipeline is not None:
            dataset = pipeline.transform_dataset(dataset)
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            split_gen = cv.split(dataset.spectra, dataset.targets)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            split_gen = cv.split(dataset.spectra)
        fold_indices = list(split_gen)
        for model_config in model_configs:
            model_name = model_config["name"]
            pbar.set_description(f"{dataset_name} | {model_name}")
            all_predictions = []
            for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
                X_train, X_test = dataset.spectra[train_idx], dataset.spectra[test_idx]
                y_train, y_test = dataset.targets[train_idx], dataset.targets[test_idx]
                model = create_model(model_config)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                for i, idx in enumerate(test_idx):
                    pred_entry = {
                        "sample_idx": idx,
                        "fold": fold_idx,
                        "y_true": y_test[i],
                        "y_pred": y_pred[i],
                    }
                    if y_proba is not None:
                        if hasattr(y_proba, 'ndim') and y_proba.ndim == 2:
                            for c in range(y_proba.shape[1]):
                                pred_entry[f"proba_class_{c}"] = y_proba[i, c]
                        else:
                            pred_entry["proba"] = y_proba[i]
                    all_predictions.append(pred_entry)
            pred_df = pd.DataFrame(all_predictions)
            safe_dataset = dataset_name.replace("/", "_").replace("\\", "_")
            filename = f"{safe_dataset}_{model_name}_predictions.csv"
            pred_df.to_csv(os.path.join(predictions_dir, filename), index=False)
    pbar.update(1)
    pbar.close()
