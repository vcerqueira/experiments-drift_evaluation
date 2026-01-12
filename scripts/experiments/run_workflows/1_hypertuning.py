"""
Hyperparameter Tuning for Drift Detectors based on Leave-one-Dataset out

This script performs hyperparameter optimization for drift detectors.
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
from sklearn.model_selection import ParameterSampler

from src.eval_detector import EvaluateDriftDetector
from src.streams.read_from_df import StreamFromDF
from src.streams.inject_drift import DriftSimulator
from src.prequential_workflow import SupervisedStreamingWorkflow
from src.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_PARAM_SPACE

from src.streams.config import MAX_DELAY, DRIFT_WIDTH

# configs
USE_PERFORMANCE_WINDOW = False
N_DRIFTS = 50
MODE = 'GRADUAL'
N_ITER_RANDOM_SEARCH = 30
RANDOM_SEED = 32
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / 'assets' / 'results'
DRIFT_REGION = (0.5, 0.8)
MIN_TRAINING_RATIO = 0.25

dataset_list = [*MAX_DELAY]

# Drift configuration parameters
DRIFT_PARAMS = [
    {'on_x_permute': True, 'on_x_exceed': False, 'on_y_prior': False, 'on_y_swap': False},
    {'on_x_permute': False, 'on_x_exceed': True, 'on_y_prior': False, 'on_y_swap': False},
    {'on_x_permute': False, 'on_x_exceed': False, 'on_y_prior': True, 'on_y_swap': False},
    {'on_x_permute': False, 'on_x_exceed': False, 'on_y_prior': False, 'on_y_swap': True},
]


def run_experiment(detector_name: str,
                   detector,
                   config: Dict[str, Any],
                   classifier_name: str,
                   dataset_name: str) -> Dict[str, Any]:
    """
    Run Monte Carlo trials experiment with the given configuration.
    
    Args:
        detector_name: Name of the drift detector
        detector: Detector class object
        config: Parameter configuration for the detector
        classifier_name: Name of the classifier to use
        dataset_name: Name of the synthetic data generator
        
    Returns:
        Dict containing the aggregated results across all trials
    """
    print(f"Running: {detector_name} with {classifier_name} on {dataset_name}")
    print(f"Config: {config}")

    np.random.seed(RANDOM_SEED)

    stream_dummy = StreamFromDF.read_stream(stream_name=dataset_name, as_np_stream=False, shuffle=False)
    stream_length = stream_dummy.shape[0]

    drift_episodes = []
    for i in range(N_DRIFTS):
        print(f'Trial {i + 1}/{N_DRIFTS}')

        stream = StreamFromDF.read_stream(stream_name=dataset_name)
        schema = stream.get_schema()

        drift_param = np.random.choice(DRIFT_PARAMS)
        drift_param['width'] = DRIFT_WIDTH[dataset_name] if MODE == 'GRADUAL' else 0

        drift_sim = DriftSimulator(
            **drift_param,
            drift_region=DRIFT_REGION,
            burn_in_samples=0,
            schema=schema
        )

        drift_sim.fit(stream_size=stream_length)
        drift_loc = drift_sim.fitted['drift_onset']
        print(f'Drift location: {drift_loc}')

        evaluator = ClassificationEvaluator(schema=schema, window_size=1)
        learner = CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name])

        trial_config = config.copy()

        # for STUDD detector, initialize and pass the student model
        if detector_name == 'STUDD':
            student = CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name])
            trial_config['student'] = student

        detector_instance = detector(**trial_config)

        wf = SupervisedStreamingWorkflow(
            model=learner,
            evaluator=evaluator,
            detector=detector_instance,
            use_window_perf=USE_PERFORMANCE_WINDOW,
            min_training_size=int(stream_length * MIN_TRAINING_RATIO),
            drift_simulator=drift_sim
        )

        monitor_instance = detector_name == 'ABCDx'

        wf.run_prequential(
            stream=stream,
            monitor_instance=monitor_instance,
            max_size=stream_length
        )

        drift_episodes.append({
            'preds': wf.drift_predictions,
            'true': (drift_loc, drift_loc)
        })

    # Evaluate performance across all drift episodes
    drift_eval = EvaluateDriftDetector(max_delay=MAX_DELAY[dataset_name])
    metrics = drift_eval.calc_performance(
        trues=None,
        preds=None,
        drift_episodes=drift_episodes,
        tot_n_instances=stream_length
    )

    metadata = {
        'detector': detector_name,
        'stream': dataset_name,
        'learner': classifier_name,
        'drift_type': MODE,
    }

    return {**metadata, 'params': config, **metrics}


def main():
    """
    Main function to run the hyperparameter tuning process.
    """

    output_file = f'{OUTPUT_DIR}/hypertuning,{MODE}7.csv'
    print(output_file)

    performance_metrics = []

    # Iterate through detectors to tune
    for detector_name, detector in DETECTORS.items():
        print(f'Running detector: {detector_name}')

        # Filter to specific detectors if needed
        # if detector_name in ['ABCD', 'ABCDx']:
        if detector_name not in ['ABCDx']:
            continue

        config_space = ParameterSampler(
            param_distributions=DETECTOR_PARAM_SPACE[detector_name],
            n_iter=N_ITER_RANDOM_SEARCH
        )

        for config in config_space:
            for classifier_name in CLASSIFIERS:
                for dataset_name in dataset_list:
                    print(config, classifier_name, dataset_name)
                    try:
                        result = run_experiment(
                            detector_name=detector_name,
                            detector=detector,
                            config=config,
                            classifier_name=classifier_name,
                            dataset_name=dataset_name
                        )
                        performance_metrics.append(result)
                    except Exception as e:
                        print(f"Error in experiment: {e}")
                        continue

                    results_df = pd.DataFrame(performance_metrics)
                    results_df.to_csv(output_file, index=False)

    results_df = pd.DataFrame(performance_metrics)
    results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
