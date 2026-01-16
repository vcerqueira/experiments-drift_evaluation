import copy
import time
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator

from src.streams.read_from_df import StreamFromDF
from src.eval_detector import EvaluateDriftDetector
from src.streams.inject_drift import DriftSimulator, DRIFT_CONFIGS
from src.prequential_workflow import SupervisedStreamingWorkflow
from src.streams.config import MAX_DELAY, DRIFT_WIDTH
from src.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_SYNTH_PARAMS

HYPERTUNING = False
PARAM_SETUP = 'hypertuned' if HYPERTUNING else 'default'
MODE = 'GRADUAL'
# MODE = 'ABRUPT'
N_DRIFTS = 50
RANDOM_SEED = 123
DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / 'assets' / 'results' / 'real'
DRIFT_REGION = (0.5, 0.8)
MIN_TRAINING_RATIO = 0.25
MAX_N_INSTANCES = 100_000
dataset_list = [*MAX_DELAY]


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

    stream_dummy = StreamFromDF.read_stream(stream_name=dataset_name, as_np_stream=False, shuffle=False)
    stream_length = stream_dummy.shape[0]

    detector_perf, detector_preds, detector_cpu_time = {}, {}, {}
    for detector_name, detector_class in DETECTORS.items():
        print(f'Running detector: {detector_name}')
        t0 = time.time()

        np.random.seed(RANDOM_SEED)

        drift_episodes = []
        for i in range(N_DRIFTS):
            print('Iter:', i)
            stream = StreamFromDF.read_stream(stream_name=dataset_name)
            schema = stream.get_schema()

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
                drift_simulator=drift_sim
            )

            monitor_instance = detector_name == 'ABCDx'

            wf.run_prequential(stream=stream,
                               monitor_instance=monitor_instance,
                               max_size=stream_length)

            drift_episodes.append({'preds': wf.drift_predictions, 'true': (drift_loc, drift_loc)})

        t1 = time.time()
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
        detector_cpu_time[detector_name] = t1 - t0

    exp_results_df = pd.DataFrame(detector_perf).T

    exp_detections_df = pd.concat(detector_preds, axis=0).reset_index(drop=True)

    return exp_results_df, exp_detections_df, detector_cpu_time


for drift_type, drift_params_ in DRIFT_CONFIGS.items():
    print(f"Running drift type: {drift_type}")
    print(f"Drift parameters: {drift_params_}")

    for dataset_name in dataset_list:
        print(dataset_name)

        drift_width = DRIFT_WIDTH[dataset_name] if MODE == 'GRADUAL' else 0

        drift_params = copy.deepcopy(drift_params_)
        drift_params['width'] = drift_width

        results_output_file = OUTPUT_DIR / f'{dataset_name},{drift_type},{MODE},{PARAM_SETUP},results.csv'
        predictions_output_file = OUTPUT_DIR / f'{dataset_name},{drift_type},{MODE},{PARAM_SETUP},predictions.csv'
        cput_output_file = OUTPUT_DIR / f'{dataset_name},{drift_type},{MODE},{PARAM_SETUP},cpu.csv'

        if os.path.exists(results_output_file):
            continue

        results_df, detections_df, cpu_time = run_experiment(
            dataset_name=dataset_name,
            classifier_name='HoeffdingTree',
            drift_type=drift_type,
            drift_params=drift_params
        )

        cpu_time_df = pd.DataFrame(pd.Series(cpu_time, name='CPU_Time'))

        results_df.to_csv(results_output_file)
        detections_df.to_csv(predictions_output_file)
        cpu_time_df.to_csv(cput_output_file)
