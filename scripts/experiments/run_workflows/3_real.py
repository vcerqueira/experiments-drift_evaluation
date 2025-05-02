import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator

from src.eval_detector import EvaluateDriftDetector
from src.streams.inject_drift import DriftSimulator
from src.prequential_workflow import SupervisedStreamingWorkflow
from src.streams.real import CAPYMOA_DATASETS, MAX_DELAY
from src.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_SYNTH_PARAMS

WIDTH = 0  # GRADUAL if > 0 ## 2000
HYPERTUNING = True
PARAM_SETUP = 'hypertuned' if HYPERTUNING else 'default'
MODE = 'GRADUAL' if WIDTH > 0 else 'ABRUPT'
N_DRIFTS = 50
RANDOM_SEED = 12
DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / 'assets' / 'results' / 'real'
DRIFT_REGION = (0.6, 0.9)
MIN_TRAINING_RATIO = 0.5
MAX_N_INSTANCES = 100_000

DRIFT_CONFIGS = {
    'x_permutations': {'width': WIDTH, 'on_x_permute': True, 'on_x_exceed': False, 'on_y_prior': False,
                       'on_y_swap': False},
    'y_swaps': {'width': WIDTH, 'on_x_permute': False, 'on_x_exceed': False, 'on_y_prior': False, 'on_y_swap': True},
    'y_prior_skip': {'width': WIDTH, 'on_x_permute': False, 'on_x_exceed': False, 'on_y_prior': True,
                     'on_y_swap': False},
    'x_exceed_skip': {'width': WIDTH, 'on_x_permute': False, 'on_x_exceed': True, 'on_y_prior': False,
                      'on_y_swap': False},
}


def run_experiment(dataset_name, classifier_name, drift_type, drift_params):
    """
    Run experiment for a specific dataset, classifier, and drift type.
    
    Args:
        dataset_name: Name of the dataset to use
        classifier_name: Name of the classifier to use
        drift_type: Type of drift to simulate
        drift_params: Parameters for the drift simulator
        
    Returns:
        DataFrame with detector performance metrics
    """
    print(f"Running experiment: {dataset_name}, {classifier_name}, {drift_type}")

    pre_stream = CAPYMOA_DATASETS[dataset_name]()
    schema = pre_stream.get_schema()
    stream_length = pre_stream._length
    stream_length = min(stream_length, MAX_N_INSTANCES)
    print('Stream sample size:', stream_length)

    detector_perf, detector_preds = {}, {}
    for detector_name, detector_class in DETECTORS.items():
        print(f'Running detector: {detector_name}')

        np.random.seed(RANDOM_SEED)

        drift_episodes = []
        for i in range(N_DRIFTS):
            print('Iter:', i)
            if dataset_name == 'Covtype':
                print('Loading dataset from csv')
                stream_df = pd.read_csv(f'{DATA_DIR}/{dataset_name}-df.csv')
                stream = DriftSimulator.shuffle_df_stream(stream_df,
                                                          dataset_name=dataset_name,
                                                          max_n_instances=MAX_N_INSTANCES)
            else:
                stream = CAPYMOA_DATASETS[dataset_name]()
                stream = DriftSimulator.shuffle_stream(stream, max_n_instances=MAX_N_INSTANCES)

            drift_sim = DriftSimulator(
                **drift_params,
                drift_region=DRIFT_REGION,
                burn_in_samples=0,
                schema=schema
            )

            drift_sim.fit(stream_size=stream_length)
            drift_loc = drift_sim.fitted['drift_onset']
            print('Drift loc:', drift_loc)

            evaluator = ClassificationEvaluator(schema=schema, window_size=1)
            learner = CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name])
            student = CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name])

            if HYPERTUNING:
                detector_params = DETECTOR_SYNTH_PARAMS[MODE]['ALL'][detector_name]
            else:
                detector_params = {}

            if detector_name == 'STUDD':
                detector_instance = detector_class(student=student, **detector_params)
            else:
                detector_instance = detector_class(**detector_params)

            wf = SupervisedStreamingWorkflow(
                model=learner,
                evaluator=evaluator,
                detector=detector_instance,
                use_window_perf=False,
                min_training_size=int(stream_length * MIN_TRAINING_RATIO),
                start_detector_on_onset=False,
                drift_simulator=drift_sim
            )

            monitor_instance = detector_name == 'ABCDx'

            wf.run_prequential(stream=stream,
                               monitor_instance=monitor_instance,
                               max_size=stream_length)

            drift_episodes.append({'preds': wf.drift_predictions, 'true': (drift_loc, drift_loc)})

        drift_eval = EvaluateDriftDetector(max_delay=MAX_DELAY[dataset_name])
        metrics = drift_eval.calc_performance(
            trues=None,
            preds=None,
            drift_episodes=drift_episodes,
            tot_n_instances=stream_length
        )

        detector_preds[detector_name] = pd.DataFrame(drift_episodes).astype(str)
        detector_preds[detector_name]['detector_name'] = detector_name

        detector_perf[detector_name] = metrics

    results_df = pd.DataFrame(detector_perf).T

    detections_df = pd.concat(detector_preds, axis=0).reset_index(drop=True)

    return results_df, detections_df


for drift_type, drift_params in DRIFT_CONFIGS.items():
    print(f"Running drift type: {drift_type}")
    print(f"Drift parameters: {drift_params}")

    for classifier_name in CLASSIFIERS:
        for dataset_name in CAPYMOA_DATASETS:
            print(dataset_name)
            # stream = CAPYMOA_DATASETS[dataset_name]()
            # sch = stream.get_schema()

            results_output_file = OUTPUT_DIR / f'{dataset_name},{drift_type},{classifier_name},{MODE},{PARAM_SETUP},results.csv'
            predictions_output_file = OUTPUT_DIR / f'{dataset_name},{drift_type},{classifier_name},{MODE},{PARAM_SETUP},predictions.csv'

            if os.path.exists(results_output_file):
                continue

            results_df, detections_df = run_experiment(
                dataset_name=dataset_name,
                classifier_name=classifier_name,
                drift_type=drift_type,
                drift_params=drift_params
            )

            results_df.to_csv(results_output_file)
            detections_df.to_csv(predictions_output_file)
            print(f"Results saved to: {results_output_file}")
