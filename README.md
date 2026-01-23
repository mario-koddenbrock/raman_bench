# Raman Bench

A comprehensive benchmark evaluation framework for machine learning models on Raman spectroscopy data.

## Features

- **Unified Data Access**: Integration with `raman_data` package for standardized dataset access
- **Extensive Model Library**: 20+ models including classical ML, deep learning, and advanced tabular models
- **Default and Tuned Versions**: Each model supports both default parameters and HPO-tuned versions
- **RamanSPy Integration**: Preprocessing pipelines built on top of RamanSPy
- **Comprehensive Metrics**: Classification and regression metrics with detailed reporting
- **Publication-Ready Visualizations**: Automated plot generation for benchmark results
- **Three-Step Pipeline**: Predictions → Metrics → Plots workflow with CSV persistence

## Installation

```bash
# Clone the repository
git clone https://github.com/raman-bench/raman_bench.git
cd raman_bench

# Create conda environment
conda env create -f environment.yml
conda activate raman_bench

# Install in development mode
pip install -e .
```

## Requirements

- Python 3.11
- PyTorch >= 2.0
- scikit-learn >= 1.0
- RamanSPy >= 0.2.0
- raman_data (required for datasets)

## Quick Start

### Using the Python API

```python
from raman_bench import BenchmarkRunner, get_model, list_models

# List available models
print(list_models(task_type=TASK_TYPE.Classification))

# Create models (default and tuned versions)
model_default = get_model("randomforest")  # Default hyperparameters
model_tuned = get_model("randomforest", tuned=True)  # With HPO

# Run benchmark
runner = BenchmarkRunner(
    datasets=["adenine"],
    models=[model_default, model_tuned],
)
results = runner.run()
runner.generate_plots()
```

### Using the CLI

```bash
# Run complete benchmark
python scripts/run_benchmark.py

# Run with custom configuration
python scripts/run_benchmark.py --config configs/default.json

# Run specific steps
python scripts/run_benchmark.py --step predictions
python scripts/run_benchmark.py --step metrics
python scripts/run_benchmark.py --step plots
```

## Available Models

### Classical Machine Learning

| Model | Default | Tuned (HPO) | Classification | Regression |
|-------|---------|-------------|----------------|------------|
| Random Forest | ✅ | ✅ | ✅ | ✅ |
| Extra Trees | ✅ | ✅ | ✅ | ✅ |
| SVM / SVR | ✅ | ✅ | ✅ | ✅ |
| XGBoost | ✅ | ✅ | ✅ | ✅ |
| LightGBM | ✅ | ✅ | ✅ | ✅ |
| CatBoost | ✅ | ✅ | ✅ | ✅ |
| KNN | ✅ | ✅ | ✅ | ✅ |
| Logistic/Linear | ✅ | ✅ | ✅ | ✅ |

### Deep Learning

| Model | Default | Tuned (HPO) | Classification | Regression |
|-------|---------|-------------|----------------|------------|
| MLP | ✅ | ✅ | ✅ | ✅ |
| CNN1D | ✅ | ✅ | ✅ | ✅ |
| TorchMLP | ✅ | ✅ | ✅ | ❌ |
| RealMLP | ✅ | ✅ | ✅ | ❌ |

### Advanced / AutoML

| Model | Default | Tuned (HPO) | Classification | Regression |
|-------|---------|-------------|----------------|------------|
| AutoGluon | ✅ | N/A* | ✅ | ✅ |
| TabPFN v1 | ✅ | N/A* | ✅ | ❌ |
| TabPFN v2 | ✅ | N/A* | ✅ | ✅ |
| TabPFN v2.5 | ✅ | N/A* | ✅ | ❌ |
| EBM (InterpretML) | ✅ | ✅ | ✅ | ✅ |
| TabDPT | ✅ | N/A* | ✅ | ❌ |
| TabM | ✅ | ✅ | ✅ | ❌ |
| ModernNCA | ✅ | ✅ | ✅ | ❌ |
| xRFM | ✅ | ✅ | ✅ | ❌ |
| SAP-RPT-OSS | ✅ | N/A* | ✅ | ❌ |

*N/A: Model handles its own hyperparameter optimization internally

## HPO (Hyperparameter Optimization)

Models that support tuning use Optuna for hyperparameter optimization:

```python
# Create a tuned model
model = get_model("xgboost", tuned=True, n_trials=50)

# Training will automatically perform HPO on a validation split
model.fit(X_train, y_train)

# Access best parameters
print(model._best_params)
```

## Benchmark Pipeline

The benchmark consists of three steps:

1. **Predictions**: Train models and generate predictions (saved as CSV)
2. **Metrics**: Compute metrics from predictions (saved as CSV)
3. **Plots**: Generate visualization plots from metrics

```python
from raman_bench import BenchmarkRunner

runner = BenchmarkRunner(
    datasets=["adenine"],
    models=[...],
    cv_folds=5,
    results_dir="results",
)

# Run all steps
results = runner.run()

# Generate plots
runner.generate_plots()

# Get summary
print(runner.summary())
```

## Project Structure

```
raman_bench/
├── src/raman_bench/
│   ├── data/           # Data handling and dataset wrapper
│   ├── models/         # Model implementations
│   │   ├── base.py     # Base classes with tuning support
│   │   ├── classical.py # Classical ML models
│   │   ├── deep_learning.py # Neural network models
│   │   ├── advanced.py # TabPFN, AutoGluon, etc.
│   │   └── registry.py # Model registry
│   ├── preprocessing/  # RamanSPy-based preprocessing
│   ├── metrics/        # Evaluation metrics
│   ├── plotting/       # Visualization tools
│   ├── evaluation/     # Benchmark runner
│   └── cli.py          # Command-line interface
├── scripts/
│   └── run_benchmark.py # Main benchmark script
├── configs/            # Configuration files
├── tests/              # Unit tests
└── results/            # Output directory
```

## Configuration

Create a YAML configuration file to customize the benchmark:

```yaml
output_dir: results
random_state: 42
cv_folds: 5
preprocessing: default

classification_datasets:
  - adenine
  - OtherDataset

classification_models:
  - name: RandomForest
    class: RandomForestClassifier
    tuned: false
    params:
      n_estimators: 100

  - name: RandomForest (Tuned)
    class: RandomForestClassifier
    tuned: true
    params:
      n_trials: 50
```

## Detailed description: Dataset handling and `RamanBenchmark`

`RamanBench` ships with a small dataset utility class called `RamanBenchmark` that
encapsulates the common dataset preparation steps used by the benchmark runner.
The class lives in `src/raman_bench/benchmark/dataset.py` and is intended to
provide a reproducible, cache-friendly interface for loading many small
spectroscopy datasets and iterating over their train/test splits.

What `RamanBenchmark` does

- Discovers datasets via the `raman_data` package by task type (classification
  or regression) and selects a configurable subset.
- Loads raw dataset objects from `raman_data`, optionally applies a preprocessing
  pipeline from `raman_bench.benchmark.preprocessing`, and converts the result
  into pandas DataFrames.
- Splits each dataset into train/test using `sklearn.model_selection.train_test_split`.
- Persists the resulting splits to simple on-disk numpy cache files (`.npy`) and
  maintains a small `index.json` mapping dataset names to their target counts to
  speed up subsequent runs.

Why this is useful

Benchmarks often iterate many dataset/model combinations. Preprocessing and
conversion to DataFrames can be expensive, especially for multi-target datasets
or when the preprocessing pipeline includes smoothing / baseline correction.
By caching prepared train/test splits, `RamanBenchmark` eliminates repeated
preprocessing work and makes experiments reproducible across runs and machines.

Quick usage

```python
from raman_bench.benchmark import RamanBenchmark

# Create a benchmark helper for the two first classification datasets
bench = RamanBenchmark(n_classification=2, n_regression=0, cache_dir='.cache')

# Iterate prepared splits
for i in range(len(bench)):
    train_df, test_df = bench[i]
    # train_df and test_df are pandas DataFrames ready to feed into models
```

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## Authors

- **Mario Koddenbrock** - HTW Berlin
- **Christoph Lange** - TU Berlin

## License

MIT License - see LICENSE file for details.

## Citation

If you use Raman Bench in your research, please cite:

```bibtex
@software{raman_bench,
  author = {Koddenbrock, Mario and Lange, Christoph},
  title = {Raman Bench: A Benchmark Framework for Raman Spectroscopy},
  year = {2026},
  url = {https://github.com/raman-bench/raman_bench}
}
```
