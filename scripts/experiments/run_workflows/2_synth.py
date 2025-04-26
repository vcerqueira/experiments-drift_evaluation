"""
Evaluating Drift Detectors on Synthetic Data Streams

This script evaluates multiple drift detectors using synthetic data streams and
predefined optimal parameters from previous hyperparameter tuning.

The workflow:
1. For each classifier and synthetic data generator combination:
2. Run all drift detectors with their optimal parameters
3. Calculate performance metrics for each detector
4. Save the results for comparative analysis
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from capymoa.evaluation.evaluation import ClassificationEvaluator
from capymoa.drift.eval_detector import EvaluateDriftDetector

from utils.streams.synth import CustomDriftStream
from utils.prequential_workflow import SupervisedStreamingWorkflow
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_SYNTH_PARAMS

USE_PERFORMANCE_WINDOW = False
MAX_DELAY = 1000
N_DRIFTS = 30
DRIFT_EVERY_N = 10000
DRIFT_WIDTH = 0
MAX_STREAM_SIZE = N_DRIFTS * (DRIFT_EVERY_N + DRIFT_WIDTH + 1)
MODE = 'ABRUPT' if DRIFT_WIDTH == 0 else 'GRADUAL'
RANDOM_SEED = 123
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / 'assets' / 'results' / 'synthetic'
GENERATORS = ['Agrawal', 'SEA', 'STAGGER']


def get_drift_points(stream) -> List[Tuple[int, int]]:
    drifts = stream.get_drifts()
    return [(x.position, x.position + x.width) for x in drifts]


def run_detector(detector_name: str,
                 detector,
                 detector_config: Dict[str, Any],
                 learner,
                 student,
                 stream,
                 generator_name: str) -> Dict[str, float]:
    """
    Run a single detector on the given stream and evaluate its performance.
    
    Args:
        detector_name: Name of the drift detector
        detector: Detector class object
        detector_config: Configuration parameters for the detector
        learner: The classifier model
        student: Student model (used by some detectors like STUDD)
        stream: The data stream to process
        generator_name: Name of the data generator (for STUDD detector)
        
    Returns:
        Dictionary of performance metrics
    """
    print(f'Running detector: {detector_name}')

    if detector_name == 'STUDD':
        adwin_conf = DETECTOR_SYNTH_PARAMS[generator_name]['ADWIN']
        adwin_detector = DETECTORS['ADWIN'](**adwin_conf)
        detector_instance = detector(student=student, detector=adwin_detector, **detector_config)
    else:
        detector_instance = detector(**detector_config)

    true_drifts = get_drift_points(stream)

    schema = stream.get_schema()
    evaluator = ClassificationEvaluator(schema=schema, window_size=1)

    wf = SupervisedStreamingWorkflow(
        model=learner,
        evaluator=evaluator,
        detector=detector_instance,
        use_window_perf=USE_PERFORMANCE_WINDOW
    )

    wf.run_prequential(stream=stream, max_size=MAX_STREAM_SIZE)

    drift_eval = EvaluateDriftDetector(max_delay=MAX_DELAY)

    metrics = drift_eval.calc_performance(
        trues=true_drifts,
        preds=wf.drift_predictions,
        tot_n_instances=wf.instances_processed
    )

    return metrics


def process_generator_classifier_pair(generator_name: str, classifier_name: str) -> None:
    output_file = OUTPUT_DIR / f"{generator_name},{classifier_name},{MODE}.csv"

    if output_file.exists():
        print(f"Results already exist for {generator_name}, {classifier_name}. Skipping.")
        return

    np.random.seed(RANDOM_SEED)
    stream_creator = CustomDriftStream(
        generator=generator_name,
        n_drifts=N_DRIFTS,
        drift_every_n=DRIFT_EVERY_N,
        drift_width=DRIFT_WIDTH
    )
    stream = stream_creator.create_stream()
    schema = stream.get_schema()

    detector_perf = {}
    for detector_name, detector_class in DETECTORS.items():
        try:
            detector_config = DETECTOR_SYNTH_PARAMS[generator_name][detector_name]

            # fresh stream for each detector to ensure fair comparison
            np.random.seed(RANDOM_SEED)
            stream_creator = CustomDriftStream(
                generator=generator_name,
                n_drifts=N_DRIFTS,
                drift_every_n=DRIFT_EVERY_N,
                drift_width=DRIFT_WIDTH
            )
            fresh_stream = stream_creator.create_stream()

            metrics = run_detector(
                detector_name=detector_name,
                detector=detector_class,
                detector_config=detector_config,
                learner=CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name]),
                student=CLASSIFIERS[classifier_name](schema=schema, **CLASSIFIER_PARAMS[classifier_name]),
                stream=fresh_stream,
                generator_name=generator_name
            )
            detector_perf[detector_name] = metrics
        except Exception as e:
            print(f"Error running detector {detector_name}: {e}")
            detector_perf[detector_name] = {"error": str(e)}

    results_df = pd.DataFrame(detector_perf).T
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file)
    print(f"Results saved to: {output_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for classifier_name in CLASSIFIERS:
        print(f"Running classifier: {classifier_name}")

        for generator_name in GENERATORS:
            print(f"Running generator: {generator_name}")

            try:
                process_generator_classifier_pair(generator_name, classifier_name)
            except Exception as e:
                print(f"Error processing {generator_name}, {classifier_name}: {e}")


if __name__ == "__main__":
    main()
