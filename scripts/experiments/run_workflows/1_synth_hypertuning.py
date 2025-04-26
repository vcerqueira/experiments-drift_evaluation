"""
Hyperparameter Tuning for Drift Detectors on Synthetic Data Streams

This script performs hyperparameter optimization for drift detectors using synthetic data streams.
It evaluates different parameter configurations using a random search approach and saves the results.

The workflow:
1. For each detector, sample N parameter configurations
2. For each configuration, evaluate performance on different synthetic data streams
3. Store and analyze the results to identify optimal configurations
"""
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, Any

from capymoa.evaluation.evaluation import ClassificationEvaluator
from capymoa.drift.eval_detector import EvaluateDriftDetector
from sklearn.model_selection import ParameterSampler

from utils.streams.synth import CustomDriftStream
from utils.prequential_workflow import SupervisedStreamingWorkflow
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_PARAM_SPACE

# configs
USE_PERFORMANCE_WINDOW = False
MAX_DELAY = 1000
N_DRIFTS = 30
DRIFT_EVERY_N = 10000
DRIFT_WIDTH = 1000
MODE = 'ABRUPT' if DRIFT_WIDTH == 0 else 'GRADUAL'
N_ITER_RANDOM_SEARCH = 30
MAX_STREAM_SIZE = N_DRIFTS * (DRIFT_EVERY_N + DRIFT_WIDTH + 1)
RANDOM_SEED = 123
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / 'assets' / 'results'


def run_experiment(detector_name: str,
                   detector,
                   config: Dict[str, Any],
                   classifier_name: str,
                   generator_name: str) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    
    Args:
        detector_name: Name of the drift detector
        detector: Detector class object
        config: Parameter configuration for the detector
        classifier_name: Name of the classifier to use
        generator_name: Name of the synthetic data generator
        
    Returns:
        Dict containing the results of the experiment
    """
    print(f"Running: {detector_name} with {classifier_name} on {generator_name}")
    print(f"Config: {config}")

    np.random.seed(RANDOM_SEED)
    stream_creator = CustomDriftStream(
        generator=generator_name,
        n_drifts=N_DRIFTS,
        drift_every_n=DRIFT_EVERY_N,
        drift_width=DRIFT_WIDTH
    )

    stream = stream_creator.create_stream()
    schema = stream.get_schema()

    evaluator = ClassificationEvaluator(schema=schema, window_size=1)
    learner = CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name])

    # for STUDD detector, initialize and pass the student model
    if detector_name == 'STUDD':
        student = CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name])
        config['student'] = student

    detector_instance = detector(**config)

    wf = SupervisedStreamingWorkflow(
        model=learner,
        evaluator=evaluator,
        detector=detector_instance,
        use_window_perf=USE_PERFORMANCE_WINDOW
    )

    monitor_instance = detector_name == 'ABCDx'

    wf.run_prequential(
        stream=stream,
        max_size=MAX_STREAM_SIZE,
        monitor_instance=monitor_instance
    )

    drifts = stream.get_drifts()
    true_drifts = [(x.position, x.position + x.width) for x in drifts]

    drift_eval = EvaluateDriftDetector(max_delay=MAX_DELAY)
    metrics = drift_eval.calc_performance(
        trues=true_drifts,
        preds=wf.drift_predictions,
        tot_n_instances=wf.instances_processed
    )

    # Compile metadata and results
    metadata = {
        'detector': detector_name,
        'stream': generator_name,
        'learner': classifier_name,
        'drift_type': MODE,
    }

    return {**metadata, 'params': config, **metrics}


def main():
    """
    Main function to run the hyperparameter tuning process.
    """

    performance_metrics = []

    # Iterate through detectors to tune
    for detector_name, detector in DETECTORS.items():
        print(f'Running detector: {detector_name}')

        # Filter to specific detectors if needed
        # if detector_name in ['ABCD']:
        #     continue

        config_space = ParameterSampler(
            param_distributions=DETECTOR_PARAM_SPACE[detector_name],
            n_iter=N_ITER_RANDOM_SEARCH
        )

        for config in config_space:
            for classifier_name in CLASSIFIERS:
                for generator_name in CustomDriftStream.GENERATORS:
                    try:
                        result = run_experiment(
                            detector_name=detector_name,
                            detector=detector,
                            config=config,
                            classifier_name=classifier_name,
                            generator_name=generator_name
                        )
                        performance_metrics.append(result)
                    except Exception as e:
                        print(f"Error in experiment: {e}")
                        continue

                    results_df = pd.DataFrame(performance_metrics)
                    output_file = f'{OUTPUT_DIR}/synthetic,hypertuning,{MODE}.csv'
                    results_df.to_csv(output_file, index=False)

    results_df = pd.DataFrame(performance_metrics)
    output_file = f'{OUTPUT_DIR}/synthetic,hypertuning,{MODE}.csv'
    results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
