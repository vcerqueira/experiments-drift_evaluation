# Drift Evaluation Framework

A comprehensive framework for evaluating concept drift detection methods based on known drift locations. This repository contains experiments that benchmark different state-of-the-art and widely used drift detection approaches.

## Project Overview

The framework provides:

- Tools for evaluating concept drift detection methods against ground truth drift locations
- A drift simulation system that can generate different types of concept drift:
  - Abrupt vs gradual drift patterns
  - Covariate drift (changes in input features)
  - Target drift (changes in output variable)
- Benchmarking capabilities for comparing drift detection methods

## Setup & Dependencies

### Requirements
- Python 3.10+
- [CapyMOA library](https://capymoa.org/)

### Installation

```bash
# Clone the repository
git clone https://github.com/vcerqueira/experiments-drift_evaluation.git
cd experiments-drift_evaluation

# Install dependencies
pip install -r requirements.txt
```

For detailed version requirements of all dependencies, see `requirements.txt`.

## Usage

All scripts should be run from the repository root with `PYTHONPATH` set:

### Running the Main Benchmark

The scripts for conducting the benchmark are in the `scripts/experiments/run_workflows` folder.

#### 1. Preliminaries

Scripts `00_collect_statistics.py` and `00_store_capymoa_data.py` run preliminary analysis.
- `00_collect_statistics.py` computes summary statistics on a given dataset for reporting purposes.
- `00_store_capymoa_data.py` stores datasets in local csv files for more efficient manipulation


#### 2. Hyperparameter Optimization

- Run `1_hypertuning.py` to conduct the hyperparameter optimization of different detectors

For each data stream and for different classifiers, each detector is ran using different configurations.
This data will be used for hyperparameter selection in a leave-one-stream-out manner. 
See `src/config.py` for the configuration space of different 
detectors and their final configuration for a given stream (in both gradual and abrupt scenarios).

This runs all configured detectors across datasets and drift configurations, outputting results to `assets/results/real/`.


#### 3. Evaluating on Real-world Data Streams

- Run `3_real.py` to conduct the evaluation framework on different real-world data streams.
- `src/config.py` — Contains detector definitions, classifier parameters, hyperparameter search spaces
- `src/streams/config.py` — Dataset-specific settings (max delay, drift width, feature medians)

This process leverages the hyperparameter optimization conducted in the previous step.


#### 4. Analysis and Visualization

After running experiments, analyze results using the scripts in the folder `scripts/experiments/analysis`.
