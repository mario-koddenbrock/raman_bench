#!/usr/bin/env python
"""Quick test script to verify raman_bench package works."""

import sys
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Testing raman_bench package imports...")

    try:
        from raman_bench import list_models, get_model
        logger.info(f"\u2713 Models module imported successfully")
        models = list_models()
        logger.info(f"  Available models ({len(models)}): {models[:10]}...")
    except Exception as e:
        logger.error(f"\u2717 Models import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    try:
        from raman_bench import DataHandler, RamanDataset
        logger.info(f"\u2713 Data module imported successfully")
    except Exception as e:
        logger.error(f"\u2717 Data import failed: {e}")
        return 1

    try:
        from raman_bench import ClassificationMetrics, RegressionMetrics
        logger.info(f"\u2713 Metrics module imported successfully")
    except Exception as e:
        logger.error(f"\u2717 Metrics import failed: {e}")
        return 1

    try:
        from raman_bench import PreprocessingPipeline, get_default_pipeline
        logger.info(f"\u2713 Preprocessing module imported successfully")
    except Exception as e:
        logger.error(f"\u2717 Preprocessing import failed: {e}")
        return 1

    try:
        from raman_bench import BenchmarkPlotter
        logger.info(f"\u2713 Plotting module imported successfully")
    except Exception as e:
        logger.error(f"\u2717 Plotting import failed: {e}")
        return 1

    try:
        from raman_bench import BenchmarkRunner
        logger.info(f"\u2713 Evaluation module imported successfully")
    except Exception as e:
        logger.error(f"\u2717 Evaluation import failed: {e}")
        return 1

    logger.info("\n\u2713 All imports successful!")

    # Quick functional test
    logger.info("\nRunning quick functional tests...")
    import numpy as np

    # Test model creation (default version)
    logger.info("\n--- Testing Default Models ---")
    model = get_model("randomforest", n_estimators=10)
    logger.info(f"\u2713 Created default model: {model}")
    assert model.tuned == False, "Default model should not be tuned"

    # Test model creation (tuned version)
    logger.info("\n--- Testing Tuned Models ---")
    tuned_model = get_model("randomforest", tuned=True, n_estimators=10, n_trials=5)
    logger.info(f"\u2713 Created tuned model: {tuned_model}")
    assert tuned_model.tuned == True, "Tuned model should have tuned=True"

    # Test metrics
    logger.info("\n--- Testing Metrics ---")
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    metrics = ClassificationMetrics()
    acc = metrics.accuracy(y_true, y_pred)
    logger.info(f"✓ Computed accuracy: {acc}")

    # Test dataset creation
    logger.info("\n--- Testing Dataset ---")
    data = np.random.randn(20, 100)
    spectra = np.arange(100)
    target = np.random.choice([0, 1], 20)
    dataset = RamanDataset(name="test", data=data, spectra=spectra, target=target)
    logger.info(f"✓ Created dataset: {dataset}")

    # Test model training (default)
    logger.info("\n--- Testing Model Training (Default) ---")
    X_train = np.random.randn(50, 20)
    y_train = np.random.choice([0, 1], 50)

    model = get_model("randomforest", n_estimators=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train[:5])
    logger.info(f"✓ Default model trained and predicted: {predictions}")

    # Test model training (tuned) - quick version
    logger.info("\n--- Testing Model Training (Tuned) ---")
    tuned_model = get_model("randomforest", tuned=True, n_estimators=10, n_trials=3)
    tuned_model.fit(X_train, y_train)
    tuned_predictions = tuned_model.predict(X_train[:5])
    logger.info(f"✓ Tuned model trained and predicted: {tuned_predictions}")
    if tuned_model._best_params:
        logger.info(f"  Best params found: {tuned_model._best_params}")

    # List all available models
    logger.info("\n--- All Available Models ---")
    all_models = list_models()
    logger.info(f"Total models registered: {len(all_models)}")

    classification_models = list_models(task_type="classification")
    logger.info(f"Classification models: {len(classification_models)}")

    regression_models = list_models(task_type="regression")
    logger.info(f"Regression models: {len(regression_models)}")

    logger.info("\n" + "=" * 50)
    logger.info("All tests passed! Package is working correctly.")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())

