import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator

from src.prequential_workflow import ProxyStreamingWorkflow
from src.streams.real import CAPYMOA_DATASETS
from src.config import CLASSIFIERS, DETECTORS

RANDOM_SEED = 123
OUTPUT_DIR = Path('assets/results')
MAX_N_INSTANCES = 70_000
LEARNER = 'HoeffdingTree'
# LEARNER = 'ARF'
DATASET = 'Covtype'

VERIFICATION_DELAY = 300
SUPERVISION_PROBABILITY = 1.
BUFFER_SIZE = 5000
UPDATE_AFTER_ALARM = True
USE_WINDOW_PERF = False

detector_perf = {}
for detector_name in DETECTORS:
    # detector_name = 'ADWIN'
    print(f'Running detector: {detector_name}')

    np.random.seed(RANDOM_SEED)

    stream = CAPYMOA_DATASETS[DATASET]()
    schema = stream.get_schema()

    evaluator = ClassificationEvaluator(schema=schema)
    evaluator_detection = ClassificationEvaluator(schema=schema)

    wf = ProxyStreamingWorkflow(
        schema=schema,
        learner=LEARNER,
        detector_name=detector_name,
        evaluator=evaluator,
        evaluator_detection=evaluator_detection,
        use_window_perf=USE_WINDOW_PERF,
        update_after_alarm=UPDATE_AFTER_ALARM,
        verification_delay=VERIFICATION_DELAY,
        supervision_proba=SUPERVISION_PROBABILITY,
        buffer_size=BUFFER_SIZE,
        min_training_size=BUFFER_SIZE
    )

    monitor_instance = detector_name == 'ABCDx'

    wf.run_prequential(stream=stream,
                       max_size=MAX_N_INSTANCES,
                       monitor_instance=monitor_instance)

    scores = dict(zip(wf.evaluator.metrics_header(), wf.evaluator.metrics()))

    detector_perf[detector_name] = scores

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.DataFrame(detector_perf).T

print(df.sort_values('f1_score'))
print(df.sort_values('accuracy'))
