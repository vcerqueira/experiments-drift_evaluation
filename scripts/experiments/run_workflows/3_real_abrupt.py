import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator
from capymoa.drift.eval_detector import EvaluateDriftDetector

from utils.streams.inject_drift import DriftSimulator
from utils.prequential_workflow import SupervisedStreamingWorkflow
from utils.streams.real import CAPYMOA_DATASETS, MAX_DELAY
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_SYNTH_PARAMS

WIDTH = 0  # ABRUPT
MODE = 'ABRUPT' if WIDTH > 0 else 'GRADUAL'
N_DRIFTS = 50
RANDOM_SEED = 123
OUTPUT_DIR = Path('assets/results')
DRIFT_REGION = (0.6, 0.9)
MIN_TRAINING_RATIO = 0.5
MAX_N_INSTANCES = 100_000

DRIFT_CONFIGS = {
    'x_permutations': {'on_x_permute': True, 'on_x_exceed': False, 'on_y_prior': False, 'on_y_swap': False},
    'x_exceed_skip': {'on_x_permute': False, 'on_x_exceed': True, 'on_y_prior': False, 'on_y_swap': False},
    'y_prior_skip': {'on_x_permute': False, 'on_x_exceed': False, 'on_y_prior': True, 'on_y_swap': False},
    'y_swaps': {'on_x_permute': False, 'on_x_exceed': False, 'on_y_prior': False, 'on_y_swap': True},
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
    # stream_length = pre_stream._length
    stream_length = min(pre_stream._length, MAX_N_INSTANCES)
    print('stream_length:', stream_length)

    pre_stream = DriftSimulator.shuffle_stream(pre_stream, max_n_instances=MAX_N_INSTANCES)
    pre_stream.next_instance()
    schema = pre_stream.get_schema()

    detector_perf = {}
    for detector_name, detector_class in DETECTORS.items():
        print(f'Running detector: {detector_name}')

        np.random.seed(RANDOM_SEED)

        drift_episodes = []
        for i in range(N_DRIFTS):
            print('Iter:', i)
            stream = CAPYMOA_DATASETS[dataset_name]()
            stream = DriftSimulator.shuffle_stream(stream)

            drift_sim = DriftSimulator(
                **drift_params,
                width=WIDTH,
                drift_region=DRIFT_REGION,
                burn_in_samples=0,
                schema=schema
            )

            drift_sim.fit(stream_size=stream_length)
            drift_loc = drift_sim.fitted['drift_onset']

            evaluator = ClassificationEvaluator(schema=schema, window_size=1)
            learner = CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name])
            student = CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name])

            detector_params = DETECTOR_SYNTH_PARAMS['ALL'][detector_name]

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

            # print('finished at', wf.instances_processed)

            drift_episodes.append({'preds': wf.drift_predictions, 'true': (drift_loc, drift_loc)})

        drift_eval = EvaluateDriftDetector(max_delay=MAX_DELAY[dataset_name])
        metrics = drift_eval.calc_performance(
            trues=None,
            preds=None,
            drift_episodes=drift_episodes,
            tot_n_instances=stream_length
        )

        detector_perf[detector_name] = metrics

    return pd.DataFrame(detector_perf).T


for drift_type, drift_params in DRIFT_CONFIGS.items():
    print(f"Running drift type: {drift_type}")
    print(f"Drift parameters: {drift_params}")

    for classifier_name in CLASSIFIERS:
        for dataset_name in CAPYMOA_DATASETS:
            print(dataset_name)
            # stream = CAPYMOA_DATASETS[dataset_name]()
            # sch = stream.get_schema()

            output_file = OUTPUT_DIR.parent.parent.parent / f'{dataset_name},{drift_type},{classifier_name},{MODE}.csv'

            if os.path.exists(output_file):
                continue

            results_df = run_experiment(
                dataset_name=dataset_name,
                classifier_name=classifier_name,
                drift_type=drift_type,
                drift_params=drift_params
            )

            results_df.to_csv(output_file)
            print(f"Results saved to: {output_file}")
